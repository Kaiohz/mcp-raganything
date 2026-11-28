"""Unit tests for the xberg document reader adapter.

The xberg engine is mocked at the module boundary (``extract``); every test
goes through the real adapter error translation and page serialization.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pydantic_core
import pytest
import xberg

from domain.errors.document import DocumentReadError, UnsupportedFormatError
from domain.ports.document_reader_port import DocumentContent
from infrastructure.document_reader.xberg_adapter import (
    XbergAdapter,
    make_xberg_config,
)


def _mock_extract(
    pages: list | None, content: str = "full", chunks: list | None = None
):
    result = MagicMock()
    r0 = MagicMock()
    r0.pages = pages
    r0.content = content
    r0.chunks = chunks or []
    result.results = [r0]
    return result


def _make_page(page_number: int, content: str, tables: list | None = None) -> dict:
    return {
        "page_number": page_number,
        "content": content,
        "tables": tables or [],
        "images": [],
    }


def _write_tmp(tmp_path, name: str = "test.pdf", data: bytes = b"doc-bytes") -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


class TestXbergAdapterExtractContent:
    @patch("infrastructure.document_reader.xberg_adapter.extract")
    async def test_extract_content_returns_document_content(
        self, mock_extract: AsyncMock, tmp_path
    ) -> None:
        mock_extract.return_value = _mock_extract([_make_page(1, "Hello world")])

        adapter = XbergAdapter(ocr_mode="vlm")
        result = await adapter.extract_content(_write_tmp(tmp_path))

        assert isinstance(result, DocumentContent)
        assert result.content == [
            {"page_number": 1, "content": "Hello world", "tables": []}
        ]

    @patch("infrastructure.document_reader.xberg_adapter.extract")
    async def test_extract_content_serializes_page_tables(
        self, mock_extract: AsyncMock, tmp_path
    ) -> None:
        class FakeTable:
            def __init__(self, markdown: str) -> None:
                self.markdown = markdown

        mock_extract.return_value = _mock_extract(
            [
                _make_page(
                    1, "text with table", tables=[FakeTable("| A | B |\n|---|---|")]
                )
            ]
        )

        adapter = XbergAdapter(ocr_mode="vlm")
        result = await adapter.extract_content(_write_tmp(tmp_path))

        assert result.content == [
            {
                "page_number": 1,
                "content": "text with table",
                "tables": [{"markdown": "| A | B |\n|---|---|"}],
            }
        ]

    @patch("infrastructure.document_reader.xberg_adapter.extract")
    async def test_extract_content_uses_tesseract_mode(
        self, mock_extract: AsyncMock, tmp_path
    ) -> None:
        mock_extract.return_value = _mock_extract([_make_page(1, "ocr text")])

        adapter = XbergAdapter(ocr_mode="tesseract")
        result = await adapter.extract_content(_write_tmp(tmp_path))

        assert result.content == [
            {"page_number": 1, "content": "ocr text", "tables": []}
        ]
        mock_extract.assert_awaited_once()
        input_obj = mock_extract.call_args.args[0]
        assert input_obj.kind == xberg.ExtractInputKind.BYTES
        assert input_obj.mime_type is None

    @patch("infrastructure.document_reader.xberg_adapter.extract")
    async def test_extract_content_passes_mime_type(
        self, mock_extract: AsyncMock, tmp_path
    ) -> None:
        mock_extract.return_value = _mock_extract([_make_page(1, "text")])

        adapter = XbergAdapter(ocr_mode="tesseract")
        await adapter.extract_content(
            _write_tmp(tmp_path, "doc.bin"), mime_type="application/pdf"
        )

        mock_extract.assert_awaited_once()
        input_obj = mock_extract.call_args.args[0]
        assert input_obj.mime_type == "application/pdf"
        assert input_obj.filename == "doc.bin"

    @patch("infrastructure.document_reader.xberg_adapter.extract")
    async def test_extract_content_splits_page_markers_when_pages_is_none(
        self, mock_extract: AsyncMock, tmp_path
    ) -> None:
        """Page-flat formats answer marker-tagged content with pages=None:
        the adapter splits the markers into per-page dicts (contract kept)."""
        mock_extract.return_value = _mock_extract(
            None,
            content="<!-- PAGE 1 -->\n\nfirst page\n\n<!-- PAGE 2 -->\n\nsecond page",
        )

        adapter = XbergAdapter(ocr_mode="tesseract")
        result = await adapter.extract_content(_write_tmp(tmp_path))

        assert result.content == [
            {"page_number": 1, "content": "first page", "tables": []},
            {"page_number": 2, "content": "second page", "tables": []},
        ]

    @patch("infrastructure.document_reader.xberg_adapter.extract")
    async def test_extract_content_groups_doc_tables_by_page_when_pages_is_none(
        self, mock_extract: AsyncMock, tmp_path
    ) -> None:
        class FakeTable:
            def __init__(self, markdown: str, page_number: int) -> None:
                self.markdown = markdown
                self.page_number = page_number

        result_mock = _mock_extract(
            None,
            content="<!-- PAGE 1 -->\n\na\n\n<!-- PAGE 2 -->\n\nb",
        )
        result_mock.results[0].tables = [
            FakeTable("| A |", 2),
            FakeTable("| B |", 1),
        ]
        mock_extract.return_value = result_mock

        adapter = XbergAdapter(ocr_mode="tesseract")
        result = await adapter.extract_content(_write_tmp(tmp_path))

        assert result.content == [
            {"page_number": 1, "content": "a", "tables": [{"markdown": "| B |"}]},
            {"page_number": 2, "content": "b", "tables": [{"markdown": "| A |"}]},
        ]

    @patch("infrastructure.document_reader.xberg_adapter.extract")
    async def test_extract_content_no_markers_builds_single_page(
        self, mock_extract: AsyncMock, tmp_path
    ) -> None:
        mock_extract.return_value = _mock_extract(None, content="flat text")

        adapter = XbergAdapter(ocr_mode="tesseract")
        result = await adapter.extract_content(_write_tmp(tmp_path))

        assert result.content == [
            {"page_number": 1, "content": "flat text", "tables": []}
        ]

    @patch("infrastructure.document_reader.xberg_adapter.extract")
    async def test_extract_content_serializes_table_objects(
        self, mock_extract: AsyncMock, tmp_path
    ) -> None:
        """Regression: engine table objects must serialize to markdown dicts.

        xberg's Table objects previously broke the MCP structured-output
        contract when left unserialized.
        """

        class FakeExtractedTable:
            def __init__(self, markdown: str) -> None:
                self.markdown = markdown
                self.cells = [["A", "B"], ["1", "2"]]
                self.page_number = 1

        mock_table = FakeExtractedTable("| A | B |\n|---|---|")
        mock_extract.return_value = _mock_extract(
            [
                _make_page(
                    1,
                    "page text",
                    tables=[mock_table],
                )
            ]
        )

        adapter = XbergAdapter(ocr_mode="tesseract")
        result = await adapter.extract_content(_write_tmp(tmp_path, "test.xlsx"))

        serialized = pydantic_core.to_jsonable_python(result)
        assert serialized["content"][0]["tables"][0] == {
            "markdown": "| A | B |\n|---|---|"
        }


class TestXbergAdapterErrorTranslation:
    @patch("infrastructure.document_reader.xberg_adapter.extract")
    async def test_parsing_error_raises_unsupported_format(
        self, mock_extract: AsyncMock, tmp_path
    ) -> None:
        from xberg import ParsingError

        mock_extract.side_effect = ParsingError("unsupported format")
        adapter = XbergAdapter(ocr_mode="vlm")

        with pytest.raises(UnsupportedFormatError, match="Unsupported file format"):
            await adapter.extract_content(_write_tmp(tmp_path, "test.xyz"))

    @patch("infrastructure.document_reader.xberg_adapter.extract")
    async def test_validation_error_raises_document_read_error(
        self, mock_extract: AsyncMock, tmp_path
    ) -> None:
        from xberg import ValidationError

        mock_extract.side_effect = ValidationError("invalid file")
        adapter = XbergAdapter(ocr_mode="vlm")

        with pytest.raises(DocumentReadError, match="Invalid file"):
            await adapter.extract_content(_write_tmp(tmp_path, "test.bad"))

    @patch("infrastructure.document_reader.xberg_adapter.extract")
    async def test_timeout_error_raises_document_read_error(
        self, mock_extract: AsyncMock, tmp_path
    ) -> None:
        from xberg import XbergTimeoutError

        mock_extract.side_effect = XbergTimeoutError("timed out")
        adapter = XbergAdapter(ocr_mode="vlm")

        with pytest.raises(DocumentReadError, match="timed out after"):
            await adapter.extract_content(_write_tmp(tmp_path, "big.pdf"))


class TestXbergAdapterExtractChunks:
    @patch("infrastructure.document_reader.xberg_adapter.extract")
    async def test_extract_chunks_returns_chunk_texts(
        self, mock_extract: AsyncMock, tmp_path
    ) -> None:
        chunk_a, chunk_b = MagicMock(), MagicMock()
        chunk_a.content = "first chunk"
        chunk_b.content = "second chunk"
        mock_extract.return_value = _mock_extract(None, chunks=[chunk_a, chunk_b])

        adapter = XbergAdapter(ocr_mode="tesseract")
        chunks = await adapter.extract_chunks(
            _write_tmp(tmp_path, "report.pdf"), 500, 100
        )

        assert chunks == ["first chunk", "second chunk"]
        config = mock_extract.call_args.args[1]
        assert config.chunking.max_characters == 500
        assert config.chunking.overlap == 100

    @patch("infrastructure.document_reader.xberg_adapter.extract")
    async def test_extract_chunks_falls_back_to_full_content(
        self, mock_extract: AsyncMock, tmp_path
    ) -> None:
        mock_extract.return_value = _mock_extract(None, content="whole document")

        adapter = XbergAdapter(ocr_mode="tesseract")
        chunks = await adapter.extract_chunks(_write_tmp(tmp_path, "report.md"))

        assert chunks == ["whole document"]

    @patch("infrastructure.document_reader.xberg_adapter.extract")
    async def test_extract_chunks_empty_document(
        self, mock_extract: AsyncMock, tmp_path
    ) -> None:
        mock_extract.return_value = _mock_extract(None, content="")

        adapter = XbergAdapter(ocr_mode="tesseract")
        assert await adapter.extract_chunks(_write_tmp(tmp_path, "empty.pdf")) == []


class TestXbergAdapterConfig:
    def test_make_xberg_config_enables_page_extraction_and_chunking(self) -> None:
        config = make_xberg_config(
            ocr_mode="tesseract", chunk_size=800, chunk_overlap=150
        )

        assert config.pages.extract_pages is True
        assert config.pages.insert_page_markers is True
        assert config.chunking.max_characters == 800
        assert config.chunking.overlap == 150
        assert config.extraction_timeout_secs > 0

    def test_default_ocr_mode_comes_from_settings(self) -> None:
        config = make_xberg_config()
        assert config.ocr is not None


class TestPageDictCoercion:
    """Both xberg page shapes normalize to the page-dict contract."""

    def test_dict_page_reads_keys_and_defaults(self) -> None:
        from infrastructure.document_reader.xberg_adapter import XbergAdapter

        assert XbergAdapter._page_dict(
            {"page_number": 3, "content": "body", "tables": []}
        ) == {"page_number": 3, "content": "body", "tables": []}
        assert XbergAdapter._page_dict({}) == {
            "page_number": 1,
            "content": "",
            "tables": [],
        }

    def test_object_page_reads_attributes_and_defaults(self) -> None:
        from infrastructure.document_reader.xberg_adapter import XbergAdapter

        class FakePage:
            page_number = 2
            content = "body"

        assert XbergAdapter._page_dict(FakePage()) == {
            "page_number": 2,
            "content": "body",
            "tables": [],
        }
        assert XbergAdapter._page_dict(object()) == {
            "page_number": 1,
            "content": "",
            "tables": [],
        }

    def test_dict_page_serializes_table_dicts(self) -> None:
        from infrastructure.document_reader.xberg_adapter import XbergAdapter

        page = {"content": "x", "tables": [{"markdown": "| A |"}, {}]}
        assert XbergAdapter._page_dict(page)["tables"] == [
            {"markdown": "| A |"},
            {"markdown": ""},
        ]
