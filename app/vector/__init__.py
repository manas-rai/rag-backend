"""Vector stores and interfaces."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorStore(ABC):
    """Interface for vector stores that handle text embeddings and similarity search."""

    @abstractmethod
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add texts to the vector store with optional metadata.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dictionaries for each text
        """

    @abstractmethod
    def add_documents(self, texts: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add documents to the vector store with optional metadata.
        
        Args:
            texts: List of texts to add
            embeddings: List of embedding vectors for each text
            metadata: Optional list of metadata dictionaries for each text
        """

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search for similar texts in the vector store.
        
        Args:
            query: The query text to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries containing text and metadata for each result
        """

    @abstractmethod
    def delete(self, texts: List[str]) -> None:
        """Delete texts from the vector store.
        
        Args:
            texts: List of texts to delete
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear all texts from the vector store."""

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the vector store.
        
        Returns:
            Dictionary containing configuration details
        """

# Import all vector store providers
from app.vector.stores.chroma_store import ChromaVectorStore
from app.vector.stores.pinecone_store import PineconeVectorStore

# Provider registry for easy access
PROVIDERS = {
    "chroma": ChromaVectorStore,
    "pinecone": PineconeVectorStore,
}

def get_provider_class(provider_name: str):
    """Get vector store provider class by name."""
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unsupported vector store provider: {provider_name}")
    return PROVIDERS[provider_name]

def get_available_providers():
    """Get list of available vector store provider names."""
    return list(PROVIDERS.keys())

# Export everything
__all__ = [
    "VectorStore",
    "ChromaVectorStore",
    "PineconeVectorStore",
    "PROVIDERS",
    "get_provider_class",
    "get_available_providers"
] 