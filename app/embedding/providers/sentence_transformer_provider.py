"""Sentence Transformer implementation of embedding provider."""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from app.embedding import EmbeddingProvider
from app.exceptions import EmbeddingError, ConfigurationError, ValidationError

class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Sentence Transformer implementation of embedding provider."""

    def __init__(
        self,
        model_name: str = "msmarco-distilbert-cos-v5",
        device: str = "cpu"
    ):
        """Initialize Sentence Transformer provider.
        
        Args:
            model_name: Name of the Sentence Transformer model
            device: Device to run the model on ('cpu' or 'cuda')
            
        Raises:
            ConfigurationError: If model initialization fails
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.device = device
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Sentence Transformer model: {str(e)}") from e

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to get embeddings for
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
            ValidationError: If input validation fails
        """
        if not texts:
            raise ValidationError("Empty text list provided")
            
        try:
            # Sentence Transformers can handle batch processing efficiently
            embeddings = self.model.encode(texts)
            
            # Convert numpy arrays to lists if needed
            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}") from e

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of the embedding vectors
            
        Raises:
            EmbeddingError: If dimension retrieval fails
        """
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception as e:
            raise EmbeddingError(f"Failed to get embedding dimension: {str(e)}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Sentence Transformers model being used.
        
        Returns:
            Dictionary containing model information
            
        Raises:
            EmbeddingError: If model info retrieval fails
        """
        try:
            return {
                "type": "sentence_transformer",
                "model": self.model_name,
                "device": self.device,
                "dimension": self.get_embedding_dimension()
            }
        except Exception as e:
            raise EmbeddingError(f"Failed to get model info: {str(e)}") from e 