"""Dependencies for the RAG Backend."""

import json
from typing import Optional
from functools import lru_cache
from fastapi import Depends, HTTPException, Request
from utils.config import get_settings, Settings

# Import base classes
from app.llm import LLMProvider
from app.embedding import EmbeddingProvider
from app.vector import VectorStore
from app.text import TextSplitter

# Import registry functions
from app.llm.registry import get_provider_class as get_llm_provider_class
from app.embedding.registry import get_provider_class as get_embedding_provider_class
from app.vector.registry import get_provider_class as get_vector_provider_class
from app.text.registry import get_splitter_class as get_text_splitter_class

# Import services
from app.document import DocumentPreProcessor
from app.services.rag_service import RAGService
from app.services.document_service import DocumentService
from app.services.query_service import QueryService

# Cache for vector store dimensions
_vector_store_dimensions = {}

def get_llm_provider(
    settings: Settings = Depends(get_settings),
    provider_type: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    deployment_name: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    model: Optional[str] = None,
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
    region: Optional[str] = None
) -> LLMProvider:
    """Get the configured LLM provider."""
    provider_type = provider_type or settings.llm_provider
    provider_class = get_llm_provider_class(provider_type)

    if provider_type == "azure":
        return provider_class(
            api_key=api_key or settings.azure_api_key,
            api_base=api_base or settings.azure_api_base,
            api_version=api_version or settings.azure_api_version,
            deployment_name=deployment_name or settings.azure_deployment_name
        )
    elif provider_type == "gcp":
        return provider_class(
            project_id=project_id or settings.gcp_project_id,
            location=location or settings.gcp_location,
            model=model or settings.gcp_model
        )
    elif provider_type == "aws":
        return provider_class(
            access_key_id=access_key_id or settings.aws_access_key_id,
            secret_access_key=secret_access_key or settings.aws_secret_access_key,
            region=region or settings.aws_region,
            model=model or settings.aws_model
        )
    elif provider_type == "groq":
        return provider_class(
            api_key=api_key or settings.groq_api_key,
            model=model or settings.groq_model
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_type}")

def get_embedding_provider(
    settings: Settings,
    config: dict,
    # provider_type: Optional[str] = None,
    # api_key: Optional[str] = None,
    # api_base: Optional[str] = None,
    # api_version: Optional[str] = None,
    # embedding_deployment_name: Optional[str] = None,
    # model: Optional[str] = None,
    # device: Optional[str] = None
) -> Optional[EmbeddingProvider]:
    """Get the configured embedding provider."""
    provider_type = config.get('embedding_provider') \
        or getattr(settings, 'embedding_provider', settings.llm_provider)

    try:
        provider_class = get_embedding_provider_class(provider_type)
    except ValueError:
        return None

    if provider_type == "azure":
        return provider_class(
            api_key=config.get('azure_api_key') or settings.azure_api_key,
            api_base=config.get('azure_api_base') or settings.azure_api_base,
            api_version=config.get('azure_api_version') or settings.azure_api_version,
            embedding_deployment_name=config.get('azure_embedding_deployment_name') \
                or getattr(settings, 'azure_embedding_deployment_name', 'text-embedding-ada-002')
        )
    elif provider_type == "groq":
        return provider_class(
            api_key=config.get('groq_api_key') or settings.groq_api_key,
            model=config.get('groq_embedding_model') \
                or getattr(settings, 'groq_embedding_model', settings.groq_model)
        )
    elif provider_type == "sentence_transformers":
        return provider_class(
            model_name=config.get('sentence_transformer_model') \
                or getattr(settings, 'sentence_transformer_model', 'all-MiniLM-L6-v2'),
            device=config.get('sentence_transformer_device') \
                or getattr(settings, 'sentence_transformer_device', 'cpu')
        )
    else:
        return None

def _get_vector_store_dimension(embedding_provider: Optional[EmbeddingProvider],
                                 settings: Settings
                                 ) -> int:
    """Get the vector store dimension, with caching."""
    cache_key = id(embedding_provider) if embedding_provider else 'default'

    if cache_key not in _vector_store_dimensions:
        if embedding_provider:
            try:
                dimension = embedding_provider.get_embedding_dimension()
            except (AttributeError, NotImplementedError):
                dimension = getattr(settings, 'vector_store_dimension', 768)
        else:
            dimension = getattr(settings, 'vector_store_dimension', 768)
        _vector_store_dimensions[cache_key] = dimension

    return _vector_store_dimensions[cache_key]

def get_vector_store(
    settings: Settings,
    config: dict,
    # provider_type: Optional[str] = None,
    # api_key: Optional[str] = None,
    # index_name: Optional[str] = None,
    # dimension: Optional[int] = None,
    # persist_directory: Optional[str] = None,
    embedding_provider: Optional[EmbeddingProvider] = Depends(get_embedding_provider)
) -> VectorStore:
    """Get the configured vector store."""
    provider_type = config.get('vector_provider') \
        or getattr(settings, 'vector_provider', 'pinecone')
    provider_class = get_vector_provider_class(provider_type)

    if provider_type == "chroma":
        return provider_class(
            persist_directory=config.get('vector_store_path') \
                or getattr(settings, 'vector_store_path', './chroma_db'),
            embedding_function=embedding_provider.get_embeddings if embedding_provider else None
        )
    elif provider_type == "pinecone":
        if not (config.get('pinecone_api_key') or settings.pinecone_api_key):
            raise ValueError("PINECONE_API_KEY is required for Pinecone vector store")

        dimension = _get_vector_store_dimension(embedding_provider, settings)

        return provider_class(
            api_key=config.get('pinecone_api_key') or settings.pinecone_api_key,
            index_name=config.get('pinecone_index_name') or settings.pinecone_index_name,
            dimension=dimension
        )
    else:
        raise ValueError(f"Unsupported vector store provider: {provider_type}")

def get_text_splitter(
    settings: Settings,
    config: dict
) -> TextSplitter:
    """Get the configured text splitter."""
    splitter_type = config.get('text_splitter', 'langchain')
    splitter_class = get_text_splitter_class(splitter_type)

    if splitter_type == "langchain":
        return splitter_class(
            chunk_size=config.get('chunk_size') or getattr(settings, 'chunk_size', 1000),
            chunk_overlap=config.get('chunk_overlap') or getattr(settings, 'chunk_overlap', 200)
        )
    elif splitter_type == "semantic":
        return splitter_class(
            chunk_size=config.get('chunk_size') or getattr(settings, 'chunk_size', 1000),
            chunk_overlap=config.get('chunk_overlap') or getattr(settings, 'chunk_overlap', 200)
        )
    else:
        raise ValueError(f"Unsupported text splitter: {splitter_type}")

def get_document_processor() -> DocumentPreProcessor:
    """Get the document processor with the configured components."""
    return DocumentPreProcessor()

@lru_cache()
def get_rag_service(
    llm_provider = Depends(get_llm_provider),
    embedding_provider = Depends(get_embedding_provider),
    vector_store = Depends(get_vector_store),
    text_splitter = Depends(get_text_splitter),
    document_processor = Depends(get_document_processor)
) -> RAGService:
    """Get the RAG service instance."""
    return RAGService(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        text_splitter=text_splitter,
        document_processor=document_processor
    )

async def get_document_service(
    request: Request,
    settings: Settings = Depends(get_settings)
) -> DocumentService:
    """Get the document service instance."""
    try:
        try:
            body = json.loads(await request.body())
            config = body.get('config', {})
        except (ValueError, AttributeError):
            config = {}

        document_processor = get_document_processor()
        text_splitter = get_text_splitter(settings=settings, config=config)
        vector_store = get_vector_store(settings=settings, config=config)
        embedding_provider = get_embedding_provider(settings=settings, config=config)

        return DocumentService(
            document_processor=document_processor,
            text_splitter=text_splitter,
            vector_store=vector_store,
            embedding_provider=embedding_provider
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@lru_cache()
def get_query_service(
    llm_provider = Depends(get_llm_provider),
    vector_store = Depends(get_vector_store),
    embedding_provider = Depends(get_embedding_provider)
) -> QueryService:
    """Get the query service instance."""
    return QueryService(
        llm_provider=llm_provider,
        vector_store=vector_store,
        embedding_provider=embedding_provider
    )
