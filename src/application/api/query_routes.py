from fastapi import APIRouter, Depends
from application.use_cases.query_use_case import QueryUseCase
from application.requests.query_request import QueryRequest
from domain.entities.query_result import QueryResult
from dependencies import get_query_use_case
from fastapi import status


query_router = APIRouter(tags=["Query"])


@query_router.post("/query", response_model=dict, status_code=status.HTTP_200_OK)
async def query_rag(
    request: QueryRequest,
    use_case: QueryUseCase = Depends(get_query_use_case),
):
    """
    Query the RAG system.

    Args:
        request: The query request containing query parameters.
        use_case: The query use case dependency.

    Returns:
        dict: Query results including answer and context.
    """
    result: QueryResult = await use_case.execute(request)
    
    # Format response based on request parameters
    if request.only_need_context:
        return {
            "chunks": result.chunks,
            "entities": result.entities or [],
            "relationships": result.relationships or [],
        }
    elif request.only_need_prompt:
        return {"prompt": result.metadata.get("prompt", "") if result.metadata else ""}
    else:
        # Standard query with answer
        response = {"result": result.answer}
        if request.include_references:
            response["chunks"] = result.chunks
            response["entities"] = result.entities or []
        return response

