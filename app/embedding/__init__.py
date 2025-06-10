"""Embedding providers and interfaces."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class EmbeddingProvider(ABC):
    """Interface for embedding providers."""

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for the given texts."""

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""

    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider."""
