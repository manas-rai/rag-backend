"""Documents router for processing and storing documents."""

from fastapi import APIRouter, Depends, HTTPException
from app.constants import ERROR_NOT_FOUND
from app.models.requests import DocumentRequest
from app.models.responses import DocumentResponse
from app.dependencies import get_document_service
from app.services.document_service import DocumentService

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={404: {"description": ERROR_NOT_FOUND}},
)

@router.post("", response_model=DocumentResponse, status_code=201)
async def process_documents(
    request: DocumentRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Process and store documents."""
    try:
        await service.process_and_store_document(
            request.documents[0].content,
            request.documents[0].metadata
        )
        return DocumentResponse(
            message="Documents processed successfully",
            document_count=len(request.documents)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process documents: {str(e)}"
        ) from e
