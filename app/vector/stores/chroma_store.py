"""ChromaDB implementation of vector store."""

import os
from typing import List, Dict, Any, Optional, Callable
from langchain_chroma import Chroma
from app.vector import VectorStore

class EmbeddingFunctionWrapper:
    """Wrapper to adapt LLM provider embedding functions to ChromaDB interface."""

    def __init__(self, embedding_function: Callable[[List[str]], List[List[float]]]):
        self.embedding_function = embedding_function

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using the wrapped function."""
        return self.embedding_function(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using the wrapped function."""
        return self.embedding_function([text])[0]

class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of vector store."""

    def __init__(
        self,
        persist_directory: str,
        embedding_function: Optional[Callable] = None
    ):
        self.persist_directory = persist_directory
        # Wrap the embedding function if provided
        if embedding_function:
            self.embedding_function = EmbeddingFunctionWrapper(embedding_function)
        else:
            self.embedding_function = None
        self._store = None
        self._initialize_store()

    def _initialize_store(self) -> None:
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

    def delete(self, texts: List[str]) -> None:
        """Delete texts from the Chroma store."""
        if self._store is None:
            return

        self._store.delete(texts)

    def clear(self) -> None:
        """Clear all texts from the Chroma store."""
        if self._store is None:
            return

        self._store.delete_collection()
        self._store = None
        self._initialize_store()

    def add_documents(self, texts: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add documents to the Chroma store."""
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the Chroma store."""
        return {
            "type": "chroma",
            "persist_directory": self.persist_directory,
            "has_embedding_function": self.embedding_function is not None
        }
