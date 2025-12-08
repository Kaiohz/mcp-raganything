from abc import ABC, abstractmethod
from typing import Optional, List
from domain.entities.document import Document


class DocumentRepoPort(ABC):
    """
    Port interface for document repository operations.
    Defines the contract for persisting document metadata.
    """

    @abstractmethod
    async def save_document(self, document: Document) -> int:
        """
        Save a document metadata to the repository.

        Args:
            document: The document entity to save.

        Returns:
            int: The ID of the saved document.
        """
        pass

    @abstractmethod
    async def get_document_by_path(self, file_path: str) -> Optional[Document]:
        """
        Retrieve a document by its file path.

        Args:
            file_path: The file path to search for.

        Returns:
            Optional[Document]: The document if found, None otherwise.
        """
        pass

    @abstractmethod
    async def list_documents(self, limit: int = 100) -> List[Document]:
        """
        List all documents in the repository.

        Args:
            limit: Maximum number of documents to return.

        Returns:
            List[Document]: List of document entities.
        """
        pass

    @abstractmethod
    async def update_document_status(
        self, file_path: str, status: str, indexed_at: int | None = None
    ) -> bool:
        """
        Update the status of a document.

        Args:
            file_path: The file path of the document to update.
            status: The new status value.
            indexed_at: Optional timestamp of when indexing completed.

        Returns:
            bool: True if update was successful, False otherwise.
        """
        pass
