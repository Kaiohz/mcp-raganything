from domain.services.indexing_service import IndexingService
from fastapi.logger import logger
import os


class IndexFileUseCase:
    """
    Use case for indexing a single file.
    Orchestrates the file indexing process.
    """

    def __init__(self, indexing_service: IndexingService) -> None:
        """
        Initialize the use case.

        Args:
            indexing_service: The service handling indexing operations.
        """
        self.indexing_service = indexing_service

    async def execute(self, file_path: str, filename: str, output_dir: str) -> dict:
        """
        Execute the file indexing process.

        Args:
            file_path: Path to the file to index.
            filename: Name of the file.
            output_dir: Output directory for processing.

        Returns:
            dict: Result message.
        """
        try:
            success = await self.indexing_service.index_file(
                file_path=file_path,
                filename=filename,
                output_dir=output_dir
            )
            
            if success:
                return {"message": f"File {filename} indexed successfully"}
            else:
                return {"error": f"Failed to index file {filename}"}
        except Exception as e:
            logger.error(f"IndexFileUseCase failed: {e}", exc_info=True)
            return {"error": str(e)}
