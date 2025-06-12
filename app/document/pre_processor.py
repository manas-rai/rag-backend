"""Document processor that coordinates text splitting, processing, and storage."""

from typing import Dict, Any, Optional
from io import BytesIO
from PyPDF2 import PdfReader
from app.utils.logger import setup_logger

logger = setup_logger('document_preprocessor')

class DocumentPreProcessor:
    """Document processor that coordinates text splitting, processing, and storage."""

    def __init__(
        self
    ):
        logger.info("Initializing DocumentPreProcessor")

    async def process_pdf_document(
        self,
        file: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process and extract text from PDF document.
        
        Args:
            file: The PDF file content as bytes
            metadata: Optional metadata for the document
            
        Returns:
            Dict containing extracted text and metadata
        """
        try:
            logger.info("Processing PDF document")
            pdf_file = BytesIO(file)
            pdf_reader = PdfReader(pdf_file)

            # Extract text from all pages
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages, 1):
                logger.debug("Extracting text from page %d", page_num)
                text_content.append(page.extract_text())

            # Combine all text
            full_text = "\n".join(text_content)
            logger.info("Successfully extracted text from %d pages", len(pdf_reader.pages))

            logger.info("Successfully processed PDF document")
            return full_text
        except Exception as e:
            logger.error("Error processing PDF document: %s", str(e))
            raise
