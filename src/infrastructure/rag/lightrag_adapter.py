from typing import Dict, Any
from domain.ports.rag_engine import RAGEnginePort
from domain.entities.query_result import QueryResult
from raganything import RAGAnything
from lightrag import QueryParam
from fastapi.logger import logger


class LightRAGAdapter(RAGEnginePort):
    """
    Adapter for RAGAnything/LightRAG implementing RAGEnginePort.
    Wraps the RAGAnything instance and provides a clean interface.
    """

    def __init__(self, rag_instance: RAGAnything) -> None:
        """
        Initialize the LightRAG adapter.

        Args:
            rag_instance: The configured RAGAnything instance.
        """
        self.rag = rag_instance
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize the RAG engine.

        Returns:
            bool: True if initialization was successful.
        """
        try:
            if not self._initialized:
                await self.rag._ensure_lightrag_initialized()
                self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}", exc_info=True)
            return False

    async def index_document(self, file_path: str, output_dir: str) -> bool:
        """
        Index a single document into the RAG system.

        Args:
            file_path: Absolute path to the document to index.
            output_dir: Directory for processing outputs.

        Returns:
            bool: True if indexing was successful.
        """
        try:
            await self.rag.process_document_complete(
                file_path=file_path,
                output_dir=output_dir,
                parse_method="auto"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to index document {file_path}: {e}", exc_info=True)
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
            folder_path: Absolute path to the folder containing documents.
            output_dir: Directory for processing outputs.
            recursive: Whether to process subdirectories recursively.
            file_extensions: Optional list of file extensions to filter.
            max_workers: Number of parallel workers for processing.

        Returns:
            Dict containing indexing results and statistics.
        """
        try:
            result = await self.rag.process_folder_complete(
                folder_path=folder_path,
                output_dir=output_dir,
                parse_method="auto",
                file_extensions=file_extensions,
                recursive=recursive,
                display_stats=True,
                max_workers=max_workers
            )
            return result
        except Exception as e:
            logger.error(f"Failed to index folder {folder_path}: {e}", exc_info=True)
            return {"error": str(e)}

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
        try:
            # Ensure initialized
            await self.initialize()
            
            # Create QueryParam from arguments
            query_param = QueryParam(
                mode=mode,
                only_need_context=only_need_context,
                only_need_prompt=kwargs.get("only_need_prompt", False),
                top_k=kwargs.get("top_k", 40),
                chunk_top_k=kwargs.get("chunk_top_k", 20),
                enable_rerank=kwargs.get("enable_rerank", True),
                include_references=kwargs.get("include_references", False),
                stream=False  # We handle streaming at FastAPI level
            )
            
            # Get context data
            result = await self.rag.lightrag.aquery_data(query, param=query_param)
            
            if result.get("status") != "success":
                error_msg = result.get("message", "Query failed")
                return QueryResult(
                    query=query,
                    answer=f"Error: {error_msg}",
                    chunks=[]
                )
            
            data = result.get("data", {})
            
            # If only context is needed, return immediately
            if only_need_context:
                return QueryResult(
                    query=query,
                    answer=None,
                    chunks=data.get("chunks", []),
                    entities=data.get("entities", []),
                    relationships=data.get("relationships", [])
                )
            
            # If only need prompt
            if kwargs.get("only_need_prompt", False):
                return QueryResult(
                    query=query,
                    answer=None,
                    chunks=[],
                    metadata={"prompt": data.get("prompt", "")}
                )
            
            # Standard query with LLM answer
            llm_result = await self.rag.lightrag.aquery(query, param=query_param)
            
            return QueryResult(
                query=query,
                answer=llm_result,
                chunks=data.get("chunks", []),
                entities=data.get("entities", []) if kwargs.get("include_references") else None,
                relationships=data.get("relationships", []) if kwargs.get("include_references") else None
            )
            
        except Exception as e:
            logger.error(f"Failed to query RAG: {e}", exc_info=True)
            return QueryResult(
                query=query,
                answer=f"Error: {str(e)}",
                chunks=[]
            )
