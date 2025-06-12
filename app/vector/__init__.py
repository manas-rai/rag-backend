"""Vector store interfaces."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

class VectorStore(ABC):
    """Interface for vector stores."""

    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    ) -> str:
        """Add texts to the vector store."""

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        query_embedding: List[float],
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar texts."""

    @abstractmethod
    def delete(self, texts: List[str]) -> None:
        """Delete texts from the vector store."""

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""

    @abstractmethod
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List documents in the vector store."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors from the store."""

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get information about the vector store."""
