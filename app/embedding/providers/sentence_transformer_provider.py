"""Sentence Transformer implementation of embedding provider."""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from app.embedding import EmbeddingProvider
from app.utils.exceptions import EmbeddingError, ConfigurationError, ValidationError
from app.utils.constants import (
    EMBEDDING_PROVIDER_TYPE_SENTENCE_TRANSFORMER,
    ERROR_EMPTY_TEXT_LIST,
    ERROR_INVALID_DEVICE
)
from app.utils.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger('sentence_transformer_embedding')

class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Sentence Transformer implementation of embedding provider."""

    def __init__(
        self,
        model_name: str = None,
        device: str = None
    ):
        """Initialize Sentence Transformer provider.
        
        Args:
            model_name: Name of the Sentence Transformer model
            device: Device to run the model on ('cpu' or 'cuda')
            
        Raises:
            ConfigurationError: If model initialization fails
            ValidationError: If device is invalid
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device

        if self.device not in ["cpu", "cuda"]:
            raise ValidationError(ERROR_INVALID_DEVICE)

        try:
            logger.info("Initializing Sentence Transformer with model: %s", self.model_name)
            self.model = SentenceTransformer(self.model_name)
            logger.info("Successfully loaded Sentence Transformer model")
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize Sentence Transformer model: {str(e)}"
                ) from e

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
            raise ValidationError(ERROR_EMPTY_TEXT_LIST)

        try:
            logger.info("Generating embeddings for %d texts", len(texts))
            # Sentence Transformers can handle batch processing efficiently
            embeddings = self.model.encode(texts)
            logger.info("Successfully generated %d embeddings", len(embeddings))

            # Convert numpy arrays to lists if needed
            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            return embeddings

        except Exception as e:
            logger.error("Error generating embeddings: %s", str(e))
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

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the Sentence Transformers model being used.
        
        Returns:
            Dictionary containing model information
            
        Raises:
            EmbeddingError: If model info retrieval fails
        """
        try:
            return {
                "type": EMBEDDING_PROVIDER_TYPE_SENTENCE_TRANSFORMER,
                "model": self.model_name,
                "device": self.device,
                "dimension": self.get_embedding_dimension()
            }
        except Exception as e:
            raise EmbeddingError(f"Failed to get model info: {str(e)}") from e
