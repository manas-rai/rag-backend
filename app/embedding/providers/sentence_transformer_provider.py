"""Sentence Transformers implementation for embedding provider interface."""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from app.embedding.interfaces import EmbeddingProvider

class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Sentence Transformers implementation of embedding provider."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """Initialize Sentence Transformers model for embeddings.
        
        Args:
            model_name: Name of the Sentence Transformers model
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using Sentence Transformers.
        
        Args:
            texts: List of texts to get embeddings for
            
        Returns:
            List of embedding vectors
        """
        # Sentence Transformers can handle batch processing efficiently
        embeddings = self.model.encode(texts, convert_to_tensor=False)

        # Convert numpy arrays to lists if needed
        if hasattr(embeddings, 'tolist'):
            return embeddings.tolist()
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            The dimension of the embedding vectors
        """
        # Get dimension from the model
        return self.model.get_sentence_embedding_dimension()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Sentence Transformers model being used."""
        return {
            "provider": "sentence_transformers",
            "embedding_model": self.model_name,
            "dimension": self.get_embedding_dimension(),
            "device": self.device,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown')
        }
