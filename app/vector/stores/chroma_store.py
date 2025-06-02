from typing import List, Dict, Any, Optional, Callable
from langchain.vectorstores import Chroma
import os
from ..interfaces import VectorStore

class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of vector store."""
    
    def __init__(
        self,
        persist_directory: str,
        embedding_function: Optional[Callable] = None
    ):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self._store = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the Chroma store."""
        if os.path.exists(self.persist_directory):
            self._store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
        else:
            self._store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add texts to the Chroma store."""
        if not texts:
            return
            
        if self._store is None:
            self._initialize_store()
            
        self._store.add_texts(texts=texts, metadatas=metadatas)
        self._store.persist()
    
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search for similar texts in the Chroma store."""
        if self._store is None:
            return []
            
        docs = self._store.similarity_search(query, k=k)
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
    
    def delete(self, ids: List[str]) -> None:
        """Delete texts from the Chroma store."""
        if self._store is None:
            return
            
        self._store.delete(ids)
        self._store.persist()
    
    def clear(self) -> None:
        """Clear all texts from the Chroma store."""
        if self._store is None:
            return
            
        self._store.delete_collection()
        self._store = None
        self._initialize_store()
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the Chroma store."""
        return {
            "type": "chroma",
            "persist_directory": self.persist_directory,
            "has_embedding_function": self.embedding_function is not None
        } 