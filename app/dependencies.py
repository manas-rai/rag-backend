from fastapi import Depends
from .config import get_settings, Settings
from .document.factory import DocumentProcessorFactory
from .llm.factory import LLMProviderFactory

def get_llm_provider(settings: Settings = Depends(get_settings)):
    """Get the configured LLM provider."""
    provider_kwargs = {}
    
    if settings.llm_provider == "openai":
        provider_kwargs = {
            "api_key": settings.openai_api_key,
            "model": settings.openai_model
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
    
    return LLMProviderFactory.create_provider(settings.llm_provider, **provider_kwargs)

def get_document_processor(settings: Settings = Depends(get_settings), llm_provider = Depends(get_llm_provider)):
    """Get the document processor with the configured components."""
    return DocumentProcessorFactory.create_default_processor(
        vector_store_path=settings.vector_store_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_function=llm_provider.get_embeddings
    ) 