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
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search for similar texts in the vector store.
        
        Args:
            query: The query text to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries containing text and metadata for each result
        """
        pass
    
    @abstractmethod
    def delete(self, texts: List[str]) -> None:
        """Delete texts from the vector store.
        
        Args:
            texts: List of texts to delete
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all texts from the vector store."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the vector store.
        
        Returns:
            Dictionary containing configuration details
        """
        pass 