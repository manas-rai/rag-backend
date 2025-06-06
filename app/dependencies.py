"""Dependencies for the RAG Backend."""

from fastapi import Depends
from app.config import get_settings, Settings
from app.document.factory import DocumentProcessorFactory
from app.llm.factory import LLMProviderFactory
from app.llm.interfaces import LLMProvider
from app.embedding.factory import EmbeddingFactory
from app.embedding.interfaces import EmbeddingProvider
from app.document.processor import DocumentProcessor

def get_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    """Get the configured LLM provider."""
    provider_kwargs = {}

    if settings.llm_provider == "azure":
        provider_kwargs = {
            "api_key": settings.azure_api_key,
            "api_base": settings.azure_api_base,
            "api_version": settings.azure_api_version,
            "deployment_name": settings.azure_deployment_name
        }
    elif settings.llm_provider == "gcp":
        provider_kwargs = {
            "project_id": settings.gcp_project_id,
            "location": settings.gcp_location,
            "model": settings.gcp_model
        }
    elif settings.llm_provider == "aws":
        provider_kwargs = {
            "access_key_id": settings.aws_access_key_id,
            "secret_access_key": settings.aws_secret_access_key,
            "region": settings.aws_region,
            "model": settings.aws_model
        }
    elif settings.llm_provider == "groq":
        provider_kwargs = {
            "api_key": settings.groq_api_key,
            "model": settings.groq_model
        }

    return LLMProviderFactory.create_provider(settings.llm_provider, **provider_kwargs)

def get_embedding_provider(settings: Settings = Depends(get_settings)) -> EmbeddingProvider:
    """Get the configured embedding provider."""
    return EmbeddingFactory.create_embedding_provider(settings)

def get_document_processor(
    settings: Settings = Depends(get_settings),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider)
) -> DocumentProcessor:
    """Get the document processor with the configured components."""
    return DocumentProcessorFactory.create_default_processor(
        vector_store_path=settings.vector_store_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_function=embedding_provider.get_embeddings if embedding_provider else None
    )
