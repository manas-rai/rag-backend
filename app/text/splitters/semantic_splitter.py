"""Semantic text splitter implementation using LangChain's SemanticChunker."""

from typing import List, Dict, Any
from langchain_experimental.text_splitter import SemanticChunker
from app.text import TextSplitter
from app.utils.logger import setup_logger

logger = setup_logger('semantic_splitter')

class SemanticTextSplitter(TextSplitter):
    """Semantic text splitter that splits text based on semantic meaning."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_function = None
    ):
        """Initialize the semantic text splitter.
        
        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            embedding_function: Function to generate embeddings for semantic splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_function = embedding_function

        self.splitter = SemanticChunker(
            embeddings=embedding_function,
            buffer_size=chunk_size,
            add_start_index=True,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0.5,
            number_of_chunks=None,
            sentence_split_regex=r"(?<=[.?!])\s+",
            min_chunk_size=chunk_size
        )
        logger.info("""Initialized semantic text splitter with chunk size: %d,
                    chunk overlap: %d, embedding function: %s""",
                    chunk_size, chunk_overlap, embedding_function)

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks based on semantic meaning.
        
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
            logger.info("Getting config for semantic splitter")
            return {
                "type": "semantic",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "has_embedding_function": self.embedding_function is not None
            }
        except Exception as e:
            logger.error("Failed to get config: %s", str(e))
            raise e
