"""Query router for processing and storing documents."""

from fastapi import APIRouter, Depends
from app.models.requests import QueryRequest
from app.models.responses import QueryResponse
from app.utils.dependencies import get_query_service
from app.services.query_service import QueryService

router = APIRouter(
    prefix="/query",
    tags=["query"],
    responses={404: {"description": "Not found"}},
)

@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service)
):
    """Query the RAG system."""
    try:
        # Get relevant chunks
        params = request.params or {}
        top_k = getattr(params, 'top_k', 4)
        temperature = getattr(params, 'temperature', 0.7)
        max_tokens = getattr(params, 'max_tokens', 500)

        response = await query_service.query_vector(
            query_text=request.query,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return QueryResponse(
            answer=response.get("response", ""),
            sources=response.get("sources", []),
            model_used=response.get("model_used", ""),
            message="Query processed successfully",
            success=True
        )
    except Exception as e:
        return QueryResponse(
            success=False,
            error=str(e),
            message="Failed to process query",
            answer="",
            sources=[],
            model_used=""
        )
