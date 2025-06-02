"""Factory for creating embedding providers."""

from typing import Optional
from app.config import Settings
from app.embedding.interfaces import EmbeddingProvider
from app.embedding.providers.azure_provider import AzureEmbeddingProvider
from app.embedding.providers.groq_provider import GroqEmbeddingProvider
from app.embedding.providers.sentence_transformer_provider import (
    SentenceTransformerEmbeddingProvider
)

class EmbeddingFactory:
    """Factory for creating embedding providers based on configuration."""

    @staticmethod
    def create_embedding_provider(settings: Settings) -> Optional[EmbeddingProvider]:
        """Create an embedding provider based on the settings.
        
        Args:
            settings: Application settings
            
        Returns:
            An embedding provider instance or None if not configured
        """
        if settings.embedding_provider == "azure":
            if not all([
                settings.azure_openai_api_key,
                settings.azure_openai_api_base,
                settings.azure_openai_api_version,
                settings.azure_openai_embedding_deployment_name
            ]):
                return None

            return AzureEmbeddingProvider(
                api_key=settings.azure_openai_api_key,
                api_base=settings.azure_openai_api_base,
                api_version=settings.azure_openai_api_version,
                embedding_deployment_name=settings.azure_openai_embedding_deployment_name
            )

        elif settings.embedding_provider == "openai":
            if not settings.openai_api_key:
                return None

        elif settings.embedding_provider == "groq":
            if not settings.groq_api_key:
                return None

            return GroqEmbeddingProvider(
                api_key=settings.groq_api_key,
                model=getattr(settings, 'groq_embedding_model', 'text-embedding-ada-002')
            )

        elif settings.embedding_provider == "sentence_transformers":
            # Sentence Transformers doesn't require API keys
            model_name = getattr(settings, 'sentence_transformer_model', 'all-MiniLM-L6-v2')
            device = getattr(settings, 'sentence_transformer_device', 'cpu')

            return SentenceTransformerEmbeddingProvider(
                model_name=model_name,
                device=device
            )

        return None
