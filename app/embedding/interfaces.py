"""Interfaces for embedding providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class EmbeddingProvider(ABC):
    """Interface for embedding providers."""

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for the given texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors (one per input text)
        """

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            The dimension of the embedding vectors
        """

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model being used.
        
        Returns:
            Dictionary containing model information
        """
