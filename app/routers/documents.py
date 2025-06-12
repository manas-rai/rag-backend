"""Documents router for processing and storing documents."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from app.utils.constants import ERROR_NOT_FOUND
from app.models.requests import DocumentRequest, PDFFile
from app.models.responses import DocumentResponse
from app.utils.dependencies import get_document_service
from app.services.document_service import DocumentService
from app.utils.logger import setup_logger

# Set up logger
logger = setup_logger('documents')

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
        logger.info("Processing %d documents", len(request.documents))
        await service.process_and_store_document(
            request.documents[0].content,
            request.documents[0].metadata
        )
        logger.info("Documents processed successfully")
        return DocumentResponse(
            message="Documents processed successfully",
            document_count=len(request.documents)
        )
    except Exception as e:
        logger.error("Failed to process documents: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process documents: {str(e)}"
        ) from e

@router.post("/upload")
async def upload_document(
    pdf_file: PDFFile = File(...),
    document_service: DocumentService = Depends(get_document_service)
):
    """Upload and process a document."""
    try:
        logger.info("Processing document: %s", pdf_file.file.filename)

        # Read file content as binary
        content = await pdf_file.file.read()
        if not content:
            logger.error("Empty file received")
            raise HTTPException(
                status_code=400,
                detail="Empty file received"
            )

        # Process the file
        result = await document_service.process_pdf_and_store(content)
        logger.info("Successfully processed document: %s", pdf_file.file.filename)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing document %s: %s", pdf_file.file.filename, str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.get("/list")
async def list_documents(
    document_service: DocumentService = Depends(get_document_service)
):
    """List all processed documents."""
    try:
        logger.info("Retrieving list of documents")
        documents = await document_service.list_documents()
        logger.info("Found %d documents", len(documents))
        return documents
    except Exception as e:
        logger.error("Error listing documents: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service)
):
    """Delete a document."""
    try:
        logger.info("Deleting document: %s", document_id)
        await document_service.delete_document(document_id)
        logger.info("Successfully deleted document: %s", document_id)
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error("Error deleting document %s: %s", document_id, str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
