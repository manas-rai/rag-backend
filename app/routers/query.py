"""Query router for processing and storing documents."""

from fastapi import APIRouter, Depends
from app.models.requests import QueryRequest
from app.models.responses import QueryResponse
from app.utils.dependencies import get_document_processor, get_llm_provider
from app.document.pre_processor import DocumentPreProcessor
from app.llm import LLMProvider

router = APIRouter(
    prefix="/query",
    tags=["query"],
    responses={404: {"description": "Not found"}},
)

@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    processor: DocumentPreProcessor = Depends(get_document_processor),
    llm_provider: LLMProvider = Depends(get_llm_provider)
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
