"""Document processor that coordinates text splitting, processing, and storage."""

from typing import List, Dict, Any, Optional
from app.utils.logger import setup_logger

logger = setup_logger('document_preprocessor')

class DocumentPreProcessor:
    """Document processor that coordinates text splitting, processing, and storage."""

    def __init__(
        self
    ):
        logger.info("Initializing DocumentPreProcessor")

    def process_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Process and store documents."""
