"""Registry for embedding providers."""

from app.embedding import EmbeddingProvider
from app.embedding.providers.azure_provider import AzureEmbeddingProvider
from app.embedding.providers.groq_provider import GroqEmbeddingProvider
from app.embedding.providers.sentence_transformer_provider import (
    SentenceTransformerEmbeddingProvider
)

# Provider registry for easy access
PROVIDERS = {
    "azure": AzureEmbeddingProvider,
    "groq": GroqEmbeddingProvider,
    "sentence_transformers": SentenceTransformerEmbeddingProvider,
}

def get_provider_class(provider_name: str):
    """Get provider class by name."""
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unsupported embedding provider: {provider_name}")
    return PROVIDERS[provider_name]

def get_available_providers():
    """Get list of available provider names."""
    return list(PROVIDERS.keys())

# Export everything
__all__ = [
    "EmbeddingProvider",
    "AzureEmbeddingProvider",
    "GroqEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "PROVIDERS",
    "get_provider_class",
    "get_available_providers"
]
