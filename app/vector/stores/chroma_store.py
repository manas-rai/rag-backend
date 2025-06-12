"""ChromaDB implementation of vector store."""

import os
from typing import List, Dict, Any, Optional, Callable
from langchain_chroma import Chroma
from app.vector import VectorStore
from app.utils.logger import setup_logger

logger = setup_logger('chroma_store')

class EmbeddingFunctionWrapper:
    """Wrapper to adapt LLM provider embedding functions to ChromaDB interface."""

    def __init__(self, embedding_function: Callable[[List[str]], List[List[float]]]):
        self.embedding_function = embedding_function
        logger.info("Initialized embedding function wrapper")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using the wrapped function."""
        logger.info("Embedding documents: %s", texts)
        return self.embedding_function(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using the wrapped function."""
        logger.info("Embedding query: %s", text)
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
        logger.info("Initialized Chroma vector store")
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
        try:
            if not texts:
                return

            if self._store is None:
                self._initialize_store()

            self._store.add_texts(texts=texts, metadatas=metadatas)
            logger.info("Added texts to Chroma store: %s", texts)
        except Exception as e:
            logger.error("Failed to add texts to Chroma store: %s", str(e))
            raise e

    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search for similar texts in the Chroma store."""
        try:
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
        except Exception as e:
            logger.error("Failed to similarity search: %s", str(e))
            raise e

    def delete(self, texts: List[str]) -> None:
        """Delete texts from the Chroma store."""
        try:
            if self._store is None:
                return

            self._store.delete(texts)
            logger.info("Deleted texts from Chroma store: %s", texts)
        except Exception as e:
            logger.error("Failed to delete texts from Chroma store: %s", str(e))
            raise e

    def clear(self) -> None:
        """Clear all texts from the Chroma store."""
        try:
            if self._store is None:
                return

            self._store.delete_collection()
            self._store = None
            self._initialize_store()
            logger.info("Cleared Chroma store")
        except Exception as e:
            logger.error("Failed to clear Chroma store: %s", str(e))
            raise e

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to the Chroma store."""

    def get_info(self) -> Dict[str, Any]:
        """Get information about the Chroma store."""
        try:
            logger.info("Getting info for Chroma store")
            return {
                "type": "chroma",
                "persist_directory": self.persist_directory,
                "has_embedding_function": self.embedding_function is not None
            }
        except Exception as e:
            logger.error("Failed to get info: %s", str(e))
            raise e

