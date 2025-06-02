from fastapi import APIRouter, Depends
from app.models.requests import DocumentRequest
from app.models.responses import DocumentResponse
from app.dependencies import get_document_processor
from app.document.processor import DocumentProcessor

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={404: {"description": "Not found"}},
)

@router.post("", response_model=DocumentResponse, status_code=201)
async def process_documents(
    request: DocumentRequest,
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """Process and store documents."""
    try:
        processor.process_documents(
            [doc.content for doc in request.documents],
            [doc.metadata for doc in request.documents]
        )
        return DocumentResponse(
            message="Documents processed successfully",
            document_count=len(request.documents)
        )
    except Exception as e:
        return DocumentResponse(
            success=False,
            error=str(e),
            message="Failed to process documents",
            document_count=0
        ) 