"""Xberg adapter for document extraction (kreuzberg v4 → xberg v1.5 migration).

The OCR/extraction engine is xberg (successor of kreuzberg v4, its v4 line is
EOL). The public shape stays the same: :class:`XbergAdapter` implements
:class:`~domain.ports.document_reader_port.DocumentReaderPort` — page-level
content + serialized tables, configurable OCR backend (VLM or Tesseract) and
chunking for the classical RAG indexing. Xberg exceptions are translated into
domain errors.

Deployment note: the ``KREUZBERG_*`` env names (and the
:class:`~config.KreuzbergConfig` keys) are KEPT so the prod configmap and the
deployed rollout order do not need a coordinated rename — only the docstrings
and the engine below changed.
"""

import logging
import os
import re
import time
from typing import TYPE_CHECKING

import aiofiles

if TYPE_CHECKING:
    # xberg 1.5.4 declares ``OutputFormat`` in its .pyi stubs for static
    # checking; the compiled runtime exports it (verified via dir()).
    from xberg.options import OutputFormat
else:
    from xberg import OutputFormat


from xberg import (
    ChunkingConfig,
    ExtractInput,
    ExtractInputKind,
    ExtractionConfig,
    OcrConfig,
    PageConfig,
    ParsingError,
    PdfConfig,
    ValidationError,
    XbergTimeoutError,
    extract,
)
from xberg import (
    LlmConfig as XbergLlmConfig,
)

from domain.errors.document import DocumentReadError, UnsupportedFormatError
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage
from domain.ports.document_reader_port import (
    DocumentContent,
    DocumentReaderPort,
)

logger = logging.getLogger(__name__)

#: The xberg/mcp-SDK-level default chunking used when the caller does not
#: override (same values as the previous kreuzberg contract).
_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 200

#: Splits the ``<!-- PAGE {page_num} -->`` markers inserted by the
#: PageConfig for formats that answer page-FLAT content (xberg leaves
#: ``doc.pages`` empty for plain text/HTML while PDFs/Office get real
#: per-page objects).
_PAGE_MARKER_RE = re.compile(r"<!-- PAGE (\d+) -->")


def make_xberg_config(
    ocr_mode: str | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> ExtractionConfig:
    """Build the xberg extraction config from the deployment settings.

    Args:
        ocr_mode: ``'vlm'`` or ``'tesseract'``; ``None`` falls back to the
            ``KREUZBERG_OCR_MODE`` setting.
        chunk_size: Maximum characters per chunk (classical indexing).
        chunk_overlap: Overlap characters between adjacent chunks.

    Returns:
        The configured :class:`xberg.ExtractionConfig`.
    """
    from config import KreuzbergConfig, LLMConfig

    cfg = KreuzbergConfig()
    if ocr_mode is None:
        ocr_mode = cfg.KREUZBERG_OCR_MODE
    llm_config = LLMConfig()
    if ocr_mode == "vlm":
        ocr = OcrConfig(
            backend="vlm",
            vlm_config=XbergLlmConfig(
                model=llm_config.VISION_MODEL,
                api_key=llm_config.api_key,
                base_url=llm_config.api_base_url,
                timeout_secs=cfg.KREUZBERG_VLM_TIMEOUT,
            ),
        )
    else:
        ocr = OcrConfig(backend="tesseract")
    return ExtractionConfig(
        pages=PageConfig(
            extract_pages=True,
            insert_page_markers=True,
            marker_format="\n\n<!-- PAGE {page_num} -->\n\n",
        ),
        use_cache=True,
        # NOTE: pyo3 complex enums expose their members via getset descriptors on
        # the CLASS — `OutputFormat.markdown` is NOT a member, it unserializes as a
        # bare descriptor (json.dumps TypeError 'getset_descriptor'). Members are
        # CONSTRUCTED: OutputFormat("markdown").
        output_format=OutputFormat("markdown"),
        enable_quality_processing=cfg.KREUZBERG_ENABLE_QUALITY_PROCESSING,
        extraction_timeout_secs=cfg.KREUZBERG_EXTRACTION_TIMEOUT,
        pdf_options=PdfConfig(
            extract_images=cfg.KREUZBERG_EXTRACT_IMAGES,
            extract_metadata=True,
        ),
        ocr=ocr,
        chunking=ChunkingConfig(max_characters=chunk_size, overlap=chunk_overlap),
    )


class XbergAdapter(DocumentReaderPort):
    """Xberg-based document extraction adapter.

    Supports 90+ file formats via xberg (kreuzberg successor), with a
    configurable OCR backend (VLM or Tesseract), page-level extraction,
    table serialization and chunking for the classical RAG indexing.
    Translates xberg exceptions into domain errors.
    """

    def __init__(self, ocr_mode: str | None = None) -> None:
        self._ocr_mode = ocr_mode
        config = make_xberg_config(ocr_mode=ocr_mode)
        self._extraction_timeout = config.extraction_timeout_secs

    def _config(
        self,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ) -> ExtractionConfig:
        return make_xberg_config(
            ocr_mode=self._ocr_mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def extract_content(
        self, file_path: str, mime_type: str = ""
    ) -> DocumentContent:
        """Extract textual content and tables from a document file.

        Args:
            file_path: Path to the local file to extract from.
            mime_type: Optional MIME type hint for xberg.

        Returns:
            DocumentContent with per-page content and serialized tables.

        Raises:
            DocumentReadError: On timeout or validation error.
            UnsupportedFormatError: If the file format cannot be parsed.
        """
        doc = await self._extract(file_path, mime_type, self._config())
        content = self._serialize_pages(doc, file_path)
        return DocumentContent(content=content)

    async def extract_chunks(
        self,
        file_path: str,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ) -> list[str]:
        """Extract a document and return its text chunks.

        Used by the classical RAG indexing: the configured chunking produces
        overlapping text chunks; a document that xberg cannot split falls back
        to the single full-content text.

        Args:
            file_path: Path to the local file to extract from.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap characters between adjacent chunks.

        Returns:
            The non-empty chunk texts.

        Raises:
            DocumentReadError: On timeout or validation error.
            UnsupportedFormatError: If the file format cannot be parsed.
        """
        doc = await self._extract(
            file_path,
            "",
            self._config(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        )
        chunks = [str(chunk.content) for chunk in (doc.chunks or [])]
        if not chunks and str(doc.content or "").strip():
            chunks = [str(doc.content)]
        return chunks

    async def _extract(self, file_path: str, mime_type: str, config: ExtractionConfig):
        """Read the file, run xberg and return the single document result.

        The whole extraction (I/O + engine) is bounded by the xberg extraction
        timeout and every xberg exception shape is surfaced as the matching
        domain error.

        Args:
            file_path: Local file path to read.
            mime_type: Optional MIME type hint (empty string = auto-detect).
            config: The xberg extraction configuration.

        Returns:
            The :class:`xberg.ExtractedDocument` for the file.
        """
        async with aiofiles.open(file_path, "rb") as f:
            data = await f.read()
        input_kwargs: dict = {
            "kind": ExtractInputKind.BYTES,
            "bytes": data,
            "filename": os.path.basename(file_path),
        }
        if mime_type:
            input_kwargs["mime_type"] = mime_type
        input_obj = ExtractInput(**input_kwargs)
        try:
            start_extract = time.time()
            result = await extract(input_obj, config)
            extract_ms = (time.time() - start_extract) * 1000
            logger.info(LogMessage.KREUZBERG_EXTRACTION_DURATION, file_path, extract_ms)
        except XbergTimeoutError as e:
            self._raise_document_error(
                ErrorMessage.KREUZBERG_EXTRACTION_TIMED_OUT.format(
                    timeout=self._extraction_timeout
                ),
                e,
            )
        except ParsingError as e:
            self._raise_unsupported_format(e)
        except ValidationError as e:
            self._raise_document_error(ErrorMessage.INVALID_FILE.format(error=e), e)
        return result.results[0]

    def _raise_document_error(self, message: str, e: Exception) -> None:
        logger.exception(message)
        raise DocumentReadError(message) from e

    def _raise_unsupported_format(self, e: Exception) -> None:
        logger.exception(ErrorMessage.UNSUPPORTED_FILE_FORMAT.format(error=e))
        raise UnsupportedFormatError(
            ErrorMessage.UNSUPPORTED_FILE_FORMAT.format(error=e)
        ) from e

    @staticmethod
    def _page_dict(page: object) -> dict:
        """Read the xberg PageContent fields via explicit attribute access.

        ``PageContent`` is a pyo3 Rust class WITHOUT ``__dict__`` — ``vars()``
        raises TypeError and its ``__dict__``-style copies leak non-JSON
        ``getset_descriptor`` objects; explicit attribute reads only (dict
        pages are accepted raw).
        """
        if isinstance(page, dict):
            return {
                "page_number": int(page.get("page_number", 1) or 1),
                "content": str(page.get("content", "") or ""),
                "tables": [
                    XbergAdapter._serialize_table(t) for t in (page.get("tables") or [])
                ],
            }
        return {
            "page_number": int(getattr(page, "page_number", 1) or 1),
            "content": str(getattr(page, "content", "") or ""),
            "tables": [
                XbergAdapter._serialize_table(t)
                for t in (getattr(page, "tables", None) or [])
            ],
        }

    @staticmethod
    def _serialize_table(t: object) -> dict:
        markdown = getattr(t, "markdown", None)
        if not markdown and isinstance(t, dict):
            markdown = t.get("markdown")
        return {"markdown": str(markdown or "")}

    def _serialize_pages(self, doc: object, file_path: str) -> list[dict]:
        """Turn the xberg document into the per-page dict contract.

        PDFs/Office files answer real per-page objects (``doc.pages``);
        page-flat formats (plain text, HTML) answer marker-tagged content
        instead — the ``<!-- PAGE n -->`` markers are split into page dicts
        and the document-level tables (which carry their ``page_number``) are
        grouped onto their page.
        """
        content: list[dict] = []
        start_serialize = time.time()

        pages = getattr(doc, "pages", None)
        if pages:
            return [self._page_dict(page) for page in pages]

        full_text = str(getattr(doc, "content", "") or "")
        tables = getattr(doc, "tables", None) or []
        tables_by_page: dict[int, list] = {}
        for table in tables:
            page_number = int(getattr(table, "page_number", 1) or 1)
            tables_by_page.setdefault(page_number, []).append(
                self._serialize_table(table)
            )

        parts = _PAGE_MARKER_RE.split(full_text)
        # split yields: [preamble, "1", text1, "2", text2, ...]
        pairs = list(zip(parts[1::2], parts[2::2], strict=True))
        if not pairs:
            content.append(
                {
                    "page_number": 1,
                    "content": full_text,
                    "tables": tables_by_page.get(1, []),
                }
            )
        else:
            for _page_num, (page_num, text) in enumerate(pairs):
                content.append(
                    {
                        "page_number": int(page_num),
                        # The marker format wraps each boundary with newlines.
                        "content": text.strip(),
                        "tables": tables_by_page.get(int(page_num), []),
                    }
                )

        serialize_ms = (time.time() - start_serialize) * 1000
        logger.info(
            LogMessage.KREUZBERG_SERIALIZATION_DURATION, file_path, serialize_ms
        )
        return content
