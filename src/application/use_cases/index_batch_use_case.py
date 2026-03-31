import logging
import os
import tempfile
import shutil
from typing import List
from fastapi import UploadFile
from domain.ports.rag_engine import RAGEnginePort
from domain.entities.indexing_result import FolderIndexingResult

logger = logging.getLogger(__name__)


class IndexBatchUseCase:
    """
    Use case for indexing multiple files in batch.
    Creates a temporary staging directory, saves all uploaded files,
    indexes the folder, and cleans up.
    """

    def __init__(self, rag_engine: RAGEnginePort, output_dir: str) -> None:
        """
        Initialize the use case.

        Args:
            rag_engine: Port for RAG engine operations.
            output_dir: Output directory for processing.
        """
        self.rag_engine = rag_engine
        self.output_dir = output_dir

    async def execute(self, files: List[UploadFile]) -> FolderIndexingResult:
        """
        Execute batch indexing process.

        Args:
            files: List of uploaded files to index.

        Returns:
            FolderIndexingResult: Structured result of the batch indexing operation.
        """
        if not files:
            raise ValueError("No files provided for batch indexing")

        # Create temporary staging directory
        staged_dir = tempfile.mkdtemp(prefix="batch_upload_")
        logger.info(f"Created temporary staging directory: {staged_dir}")

        try:
            # Save all uploaded files to staging directory
            saved_files = []
            for file in files:
                if file.filename:
                    file_path = os.path.join(staged_dir, file.filename)
                    with open(file_path, "wb") as f:
                        shutil.copyfileobj(file.file, f)
                    saved_files.append(file_path)
                    logger.info(f"Saved file: {file.filename}")

            logger.info(f"Saved {len(saved_files)} files to staging directory")

            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)

            # Index the entire staging folder
            result = await self.rag_engine.index_folder(
                folder_path=staged_dir,
                output_dir=self.output_dir,
                recursive=False,  # No need for recursion in flat upload
                file_extensions=None,  # Accept all file types
            )

            logger.info(
                f"Batch indexing completed: {result.stats.files_processed} processed, "
                f"{result.stats.files_failed} failed"
            )

            return result

        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(staged_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory: {staged_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up staging directory: {e}")
