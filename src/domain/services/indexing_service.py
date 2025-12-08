from typing import Dict, Any
from domain.ports.rag_engine import RAGEnginePort
from domain.ports.document_repo import DocumentRepoPort
from domain.entities.document import Document
from fastapi.logger import logger
import time


class IndexingService:
    """
    Service for indexing documents into the RAG system.
    Orchestrates the process of indexing files and tracking their metadata.
    """

    def __init__(
        self,
        rag_engine: RAGEnginePort,
        document_repo: DocumentRepoPort,
    ) -> None:
        """
        Initialize the indexing service with required ports.

        Args:
            rag_engine: Port for RAG engine operations.
            document_repo: Port for persisting document metadata.
        """
        self.rag_engine = rag_engine
        self.document_repo = document_repo

    async def index_file(self, file_path: str, filename: str, output_dir: str) -> bool:
        """
        Index a single file and track its metadata.

        Args:
            file_path: Path to the file to index.
            filename: Name of the file.
            output_dir: Output directory for processing.

        Returns:
            bool: True if indexing was successful, False otherwise.
        """
        try:
            # Create document metadata
            document = Document(
                file_path=file_path,
                filename=filename,
                status="pending"
            )
            
            # Save initial metadata
            await self.document_repo.save_document(document)
            
            # Index the document
            success = await self.rag_engine.index_document(file_path, output_dir)
            
            # Update status
            if success:
                await self.document_repo.update_document_status(
                    file_path=file_path,
                    status="indexed",
                    indexed_at=int(time.time())
                )
            else:
                await self.document_repo.update_document_status(
                    file_path=file_path,
                    status="failed"
                )
            
            return success
        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}", exc_info=True)
            await self.document_repo.update_document_status(
                file_path=file_path,
                status="failed"
            )
            return False

    async def index_folder(
        self,
        folder_path: str,
        output_dir: str,
        recursive: bool = True,
        file_extensions: list[str] | None = None,
        max_workers: int | None = None,
    ) -> Dict[str, Any]:
        """
        Index all documents in a folder.

        Args:
            folder_path: Path to the folder to index.
            output_dir: Output directory for processing.
            recursive: Whether to process subdirectories.
            file_extensions: Optional file extension filter.
            max_workers: Number of parallel workers.

        Returns:
            Dict containing indexing results.
        """
        try:
            result = await self.rag_engine.index_folder(
                folder_path=folder_path,
                output_dir=output_dir,
                recursive=recursive,
                file_extensions=file_extensions,
                max_workers=max_workers,
            )
            return result
        except Exception as e:
            logger.error(f"Failed to index folder {folder_path}: {e}", exc_info=True)
            return {"error": str(e)}
