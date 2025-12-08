from domain.ports.rag_engine import RAGEnginePort
from domain.entities.query_result import QueryResult
from fastapi.logger import logger


class QueryService:
    """
    Service for querying the RAG system.
    Orchestrates the query process and handles result formatting.
    """

    def __init__(self, rag_engine: RAGEnginePort) -> None:
        """
        Initialize the query service with required ports.

        Args:
            rag_engine: Port for RAG engine operations.
        """
        self.rag_engine = rag_engine

    async def query(
        self,
        query: str,
        mode: str = "hybrid",
        only_need_context: bool = False,
        only_need_prompt: bool = False,
        top_k: int = 40,
        chunk_top_k: int = 20,
        enable_rerank: bool = True,
        include_references: bool = False,
    ) -> QueryResult:
        """
        Execute a query against the RAG system.

        Args:
            query: The query string.
            mode: Query mode (naive, local, global, hybrid).
            only_need_context: Return only context without LLM generation.
            only_need_prompt: Return only the constructed prompt.
            top_k: Number of top entities/relations to retrieve.
            chunk_top_k: Number of top chunks to retrieve.
            enable_rerank: Enable reranking of results.
            include_references: Include references in the response.

        Returns:
            QueryResult: The structured query result.
        """
        try:
            result = await self.rag_engine.query(
                query=query,
                mode=mode,
                only_need_context=only_need_context,
                only_need_prompt=only_need_prompt,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                enable_rerank=enable_rerank,
                include_references=include_references,
            )
            return result
        except Exception as e:
            logger.error(f"Failed to query RAG: {e}", exc_info=True)
            return QueryResult(
                query=query,
                answer=f"Error: {str(e)}",
                chunks=[],
            )
