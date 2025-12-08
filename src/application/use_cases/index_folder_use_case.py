from domain.services.indexing_service import IndexingService
from application.requests.indexing_request import IndexFolderRequest
from fastapi.logger import logger


class IndexFolderUseCase:
    """
    Use case for indexing a folder of documents.
    Orchestrates the folder indexing process.
    """

    def __init__(self, indexing_service: IndexingService) -> None:
        """
        Initialize the use case.

        Args:
            indexing_service: The service handling indexing operations.
        """
        self.indexing_service = indexing_service

    async def execute(self, request: IndexFolderRequest, output_dir: str) -> dict:
        """
        Execute the folder indexing process.

        Args:
            request: The indexing request containing folder parameters.
            output_dir: Output directory for processing.

        Returns:
            dict: Indexing results and statistics.
        """
        try:
            result = await self.indexing_service.index_folder(
                folder_path=request.folder_path,
                output_dir=output_dir,
                recursive=request.recursive,
                file_extensions=request.file_extensions,
                max_workers=request.max_workers,
            )
            
            return {
                "message": f"Successfully processed folder: {request.folder_path}",
                "result": result
            }
        except Exception as e:
            logger.error(f"IndexFolderUseCase failed: {e}", exc_info=True)
            return {"error": str(e)}
