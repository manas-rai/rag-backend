from fastapi import APIRouter, Depends
from ..models.requests import QueryRequest
from ..models.responses import QueryResponse
from ..dependencies import get_document_processor, get_llm_provider

router = APIRouter(
    prefix="/query",
    tags=["query"],
    responses={404: {"description": "Not found"}},
)

@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    processor = Depends(get_document_processor),
    llm_provider = Depends(get_llm_provider)
):
    """Query the RAG system."""
    try:
        # Get relevant chunks
        chunks = processor.get_relevant_chunks(request.query, k=request.k)
        
        # Generate response using LLM
        response = llm_provider.generate_response(
            query=request.query,
            context=chunks,
            max_tokens=request.max_tokens
        )
        
        return QueryResponse(
            response=response,
            chunks=chunks
        )
    except Exception as e:
        return QueryResponse(
            success=False,
            error=str(e),
            message="Failed to process query",
            response="",
            chunks=[]
        ) 