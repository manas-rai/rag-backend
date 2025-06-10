"""Registry for vector stores."""

from app.vector import VectorStore
from app.vector.stores.chroma_store import ChromaVectorStore
from app.vector.stores.pinecone_store import PineconeVectorStore

# Provider registry for easy access
PROVIDERS = {
    "chroma": ChromaVectorStore,
    "pinecone": PineconeVectorStore,
}

def get_provider_class(provider_name: str):
    """Get provider class by name."""
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unsupported vector store provider: {provider_name}")
    return PROVIDERS[provider_name]

def get_available_providers():
    """Get list of available provider names."""
    return list(PROVIDERS.keys())

# Export everything
__all__ = [
    "VectorStore",
    "ChromaVectorStore",
    "PineconeVectorStore",
    "PROVIDERS",
    "get_provider_class",
    "get_available_providers"
]
