import contextlib
import logging
import os
from pathlib import Path

import aiofiles
from kreuzberg import ExtractionConfig, extract_file

from application.use_cases._user_id_tag import tag_documents_with_user_id
from domain.entities.indexing_result import (
    FileProcessingDetail,
    FolderIndexingResult,
    FolderIndexingStats,
    IndexingStatus,
)
from domain.logging.messages import LogMessage
from domain.ports.storage_port import StoragePort
from domain.ports.vector_store_port import VectorStorePort
from domain.services.classical_helpers import (
    build_documents_from_extraction,
    validate_path,
)
from infrastructure.document_reader.kreuzberg_adapter import make_extraction_config

logger = logging.getLogger(__name__)


class ClassicalIndexFolderUseCase:
    """Index all files in a storage folder into the classical RAG vector store.

    Iterates over objects under a prefix, extracts and chunks each file,
    and adds the documents to the vector store. Reports per-file status
    (SUCCESS / FAILED / PARTIAL) with aggregated stats.
    """

    def __init__(
        self,
        vector_store: VectorStorePort,
        storage: StoragePort,
        bucket: str,
        output_dir: str,
    ) -> None:
        self.vector_store = vector_store
        self.storage = storage
        self.bucket = bucket
        self.output_dir = output_dir

    async def _process_file(
        self,
        working_dir: str,
        file_name: str,
        *,
        index: int,
        total: int,
        config: ExtractionConfig,
    ) -> FileProcessingDetail:
        """Extract, chunk and index one storage object; never raises.

        Args:
            working_dir: Storage prefix / vector store namespace.
            file_name: The storage object being processed.
            index: Zero-based position of this file (logs use ``index + 1``,
                matching the historical ``processed + failed`` counters).
            total: Total number of files in this batch.
            config: Kreuzberg extraction configuration.

        Returns:
            :class:`FileProcessingDetail` for this file (SUCCESS or FAILED).
        """
        local_path: str | None = None
        try:
            logger.info(
                LogMessage.CLASSICAL_FOLDER_INDEX_FILE_START,
                index + 1,
                total,
                file_name,
            )
            data = await self.storage.get_object(self.bucket, file_name)

            local_path = validate_path(self.output_dir, file_name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            async with aiofiles.open(local_path, "wb") as f:
                await f.write(data)

            result = await extract_file(local_path, config=config)

            documents = build_documents_from_extraction(result, file_name)

            documents = tag_documents_with_user_id(documents)

            if documents:
                await self.vector_store.add_documents(
                    working_dir=working_dir, documents=documents
                )

            logger.info(
                LogMessage.CLASSICAL_FOLDER_INDEX_FILE_DONE,
                index + 1,
                total,
                file_name,
                len(documents),
            )
            return FileProcessingDetail(
                file_path=file_name,
                file_name=os.path.basename(file_name),
                status=IndexingStatus.SUCCESS,
            )
        except Exception as exc:
            logger.error(
                LogMessage.CLASSICAL_FOLDER_INDEX_FILE_FAILED,
                index + 1,
                total,
                file_name,
                str(exc),
            )
            return FileProcessingDetail(
                file_path=file_name,
                file_name=os.path.basename(file_name),
                status=IndexingStatus.FAILED,
                error=str(exc),
            )
        finally:
            if local_path and os.path.exists(local_path):
                with contextlib.suppress(OSError):
                    os.remove(local_path)

    async def execute(
        self,
        working_dir: str,
        recursive: bool = True,
        file_extensions: list[str] | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> FolderIndexingResult:
        """Index all matching files under a storage prefix.

        Args:
            working_dir: Storage prefix and vector store namespace.
            recursive: Whether to traverse sub-folders.
            file_extensions: Optional filter (e.g. ``[".pdf", ".docx"]``).
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap characters between adjacent chunks.

        Returns:
            FolderIndexingResult with per-file details and aggregated stats.
        """
        logger.info(LogMessage.CLASSICAL_FOLDER_INDEX_STARTED, working_dir, recursive)
        await self.vector_store.ensure_table(working_dir)

        prefix = working_dir if working_dir.endswith("/") else f"{working_dir}/"

        files = await self.storage.list_objects(
            self.bucket, prefix=prefix, recursive=recursive
        )

        if file_extensions:
            exts = set(file_extensions)
            files = [f for f in files if Path(f).suffix in exts]

        if not files:
            logger.info(LogMessage.CLASSICAL_FOLDER_INDEX_NO_FILES, prefix)
        else:
            logger.info(LogMessage.CLASSICAL_FOLDER_INDEX_LISTING, len(files), prefix)

        processed = 0
        failed = 0
        file_results = []
        config = make_extraction_config(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        for index, file_name in enumerate(files):
            detail = await self._process_file(
                working_dir,
                file_name,
                index=index,
                total=len(files),
                config=config,
            )
            if detail.status is IndexingStatus.SUCCESS:
                processed += 1
            else:
                failed += 1
            file_results.append(detail)

        if failed > 0 and processed > 0:
            status = IndexingStatus.PARTIAL
        elif failed > 0:
            status = IndexingStatus.FAILED
        else:
            status = IndexingStatus.SUCCESS

        logger.info(
            LogMessage.CLASSICAL_FOLDER_INDEX_DONE,
            working_dir,
            len(files),
            processed,
            failed,
            status.value,
        )
        return FolderIndexingResult(
            status=status,
            message="Folder indexing completed",
            folder_path=working_dir,
            recursive=recursive,
            stats=FolderIndexingStats(
                total_files=len(files),
                files_processed=processed,
                files_failed=failed,
            ),
            file_results=file_results,
        )
