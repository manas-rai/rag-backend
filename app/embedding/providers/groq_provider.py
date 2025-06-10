"""Groq implementation for embedding provider interface."""

from typing import List, Dict, Any
from groq import Groq
from app.embedding import EmbeddingProvider

class GroqEmbeddingProvider(EmbeddingProvider):
    """Groq implementation of embedding provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002"  # Update with actual Groq embedding model
    ):
        """Initialize Groq client for embeddings.
        
        Args:
            api_key: Groq API key
            model: Groq embedding model name
        """
        self.client = Groq(api_key=api_key)
        self.model = model

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using Groq.
        
        Args:
            texts: List of texts to get embeddings for
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            # Note: Update this when Groq releases embedding API
            # For now, this is a placeholder structure
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            The dimension of the embedding vectors
        """
        # Update with actual Groq embedding model dimensions
        return 1536

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Groq embedding model being used."""
        return {
            "provider": "groq",
            "embedding_model": self.model,
            "dimension": self.get_embedding_dimension()
        }
