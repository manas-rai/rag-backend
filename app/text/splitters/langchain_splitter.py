"""LangChain implementation of text splitter."""

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.text import TextSplitter
from app.utils.logger import setup_logger

logger = setup_logger('langchain_splitter')

class LangChainTextSplitter(TextSplitter):
    """LangChain implementation of text splitter."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """Initialize the LangChain text splitter.
        
        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            separators: List of separators to use for splitting
        """
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
        logger.info("""Initialized LangChain text splitter with chunk size: %d,
                    chunk overlap: %d, separators: %s""",
                    chunk_size, chunk_overlap, separators)

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        try:
            logger.info("Splitting text: %s", text)
            return self.splitter.split_text(text)
        except Exception as e:
            logger.error("Failed to split text: %s", str(e))
            raise e

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the splitter.
        
        Returns:
            Dictionary containing splitter configuration
        """
        try:
            logger.info("Getting config for langchain splitter")
            return {
                "type": "langchain",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "separators": self.separators
            }
        except Exception as e:
            logger.error("Failed to get config: %s", str(e))
            raise e
