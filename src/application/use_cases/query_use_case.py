from domain.services.query_service import QueryService
from application.requests.query_request import QueryRequest
from domain.entities.query_result import QueryResult
from fastapi.logger import logger


class QueryUseCase:
    """
    Use case for querying the RAG system.
    Orchestrates the query process.
    """

    def __init__(self, query_service: QueryService) -> None:
        """
        Initialize the use case.

        Args:
            query_service: The service handling query operations.
        """
        self.query_service = query_service

    async def execute(self, request: QueryRequest) -> QueryResult:
        """
        Execute the query process.

        Args:
            request: The query request containing query parameters.

        Returns:
            QueryResult: The structured query result.
        """
        try:
            result = await self.query_service.query(
                query=request.query,
                mode=request.mode,
                only_need_context=request.only_need_context,
                only_need_prompt=request.only_need_prompt,
                top_k=request.top_k,
                chunk_top_k=request.chunk_top_k,
                enable_rerank=request.enable_rerank,
                include_references=request.include_references,
            )
            return result
        except Exception as e:
            logger.error(f"QueryUseCase failed: {e}", exc_info=True)
            return QueryResult(
                query=request.query,
                answer=f"Error: {str(e)}",
                chunks=[],
            )
