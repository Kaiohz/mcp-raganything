from abc import ABC, abstractmethod
from typing import Dict, Any
from domain.entities.query_result import QueryResult


class RAGEnginePort(ABC):
    """
    Port interface for RAG engine operations.
    Defines the contract for RAG indexing and querying functionality.
    """

    @abstractmethod
    async def index_document(self, file_path: str, output_dir: str) -> bool:
        """
        Index a single document into the RAG system.

        Args:
            file_path: Absolute path to the document to index.
            output_dir: Directory for processing outputs.

        Returns:
            bool: True if indexing was successful, False otherwise.
        """
        pass

    @abstractmethod
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
            folder_path: Absolute path to the folder containing documents.
            output_dir: Directory for processing outputs.
            recursive: Whether to process subdirectories recursively.
            file_extensions: Optional list of file extensions to filter.
            max_workers: Number of parallel workers for processing.

        Returns:
            Dict containing indexing results and statistics.
        """
        pass

    @abstractmethod
    async def query(
        self,
        query: str,
        mode: str = "hybrid",
        only_need_context: bool = False,
        **kwargs
    ) -> QueryResult:
        """
        Query the RAG system.

        Args:
            query: The query string.
            mode: Query mode (naive, local, global, hybrid).
            only_need_context: Return only context without LLM generation.
            **kwargs: Additional query parameters.

        Returns:
            QueryResult: The structured query result.
        """
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the RAG engine.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        pass
