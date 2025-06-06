"""LangChain implementation of text splitter."""

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.text.interfaces import TextSplitter

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

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        return self.splitter.split_text(text)

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the splitter.
        
        Returns:
            Dictionary containing splitter configuration
        """
        return {
            "type": "langchain",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": self.separators
        }
