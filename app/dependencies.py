"""Dependencies for the RAG Backend."""

from typing import Optional
from fastapi import Depends
from app.config import get_settings, Settings

# Import components through their __init__.py interfaces
from app.llm import LLMProvider, get_provider_class as get_llm_provider_class
from app.embedding import EmbeddingProvider, get_provider_class as get_embedding_provider_class
from app.vector import VectorStore, get_provider_class as get_vector_provider_class
from app.text import TextSplitter, get_splitter_class as get_text_splitter_class
from app.document import DocumentPreProcessor
from app.services.rag_service import RAGService
from app.services.document_service import DocumentService
from app.services.query_service import QueryService

def get_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    """Get the configured LLM provider."""
    provider_class = get_llm_provider_class(settings.llm_provider)

    if settings.llm_provider == "azure":
        return provider_class(
            api_key=settings.azure_api_key,
            api_base=settings.azure_api_base,
            api_version=settings.azure_api_version,
            deployment_name=settings.azure_deployment_name
        )
    elif settings.llm_provider == "gcp":
        return provider_class(
            project_id=settings.gcp_project_id,
            location=settings.gcp_location,
            model=settings.gcp_model
        )
    elif settings.llm_provider == "aws":
        return provider_class(
            access_key_id=settings.aws_access_key_id,
            secret_access_key=settings.aws_secret_access_key,
            region=settings.aws_region,
            model=settings.aws_model
        )
    elif settings.llm_provider == "groq":
        return provider_class(
            api_key=settings.groq_api_key,
            model=settings.groq_model
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

def get_embedding_provider(settings: Settings = Depends(get_settings)) -> Optional[EmbeddingProvider]:
    """Get the configured embedding provider."""
    # Determine embedding provider from settings or fallback to LLM provider
    embedding_provider_name = getattr(settings, 'embedding_provider', settings.llm_provider)

    try:
        provider_class = get_embedding_provider_class(embedding_provider_name)
    except ValueError:
        # If embedding provider not found, return None
        return None

    if embedding_provider_name == "azure":
        return provider_class(
            api_key=settings.azure_api_key,
            api_base=settings.azure_api_base,
            api_version=settings.azure_api_version,
            embedding_deployment_name=getattr(settings,
                                              'azure_embedding_deployment_name',
                                              'text-embedding-ada-002')
        )
    elif embedding_provider_name == "groq":
        return provider_class(
            api_key=settings.groq_api_key,
            model=getattr(settings, 'groq_embedding_model', settings.groq_model)
        )
    elif embedding_provider_name == "sentence_transformers":
        return provider_class(
            model_name=getattr(settings, 'sentence_transformer_model', 'all-MiniLM-L6-v2'),
            device=getattr(settings, 'sentence_transformer_device', 'cpu')
        )
    else:
        return None

def get_vector_store(
    settings: Settings = Depends(get_settings),
    embedding_provider: Optional[EmbeddingProvider] = Depends(get_embedding_provider)
) -> VectorStore:
    """Get the configured vector store."""
    # Default to chroma if not specified
    vector_provider_name = getattr(settings, 'vector_provider', 'Pinecone')
    provider_class = get_vector_provider_class(vector_provider_name)

    if vector_provider_name == "chroma":
        return provider_class(
            persist_directory=getattr(settings, 'vector_store_path', './chroma_db'),
            embedding_function=embedding_provider.get_embeddings if embedding_provider else None
        )
    elif vector_provider_name == "pinecone":
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required for Pinecone vector store")

        # Get embedding dimension from provider if available
        dimension = 768  # Default dimension
        if embedding_provider:
            try:
                dimension = embedding_provider.get_embedding_dimension()
            except:
                pass  # Use default if method not available

        return provider_class(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            dimension=dimension
        )
    else:
        raise ValueError(f"Unsupported vector store provider: {vector_provider_name}")

def get_text_splitter(settings: Settings = Depends(get_settings)) -> TextSplitter:
    """Get the configured text splitter."""
    # Default to langchain if not specified
    splitter_name = getattr(settings, 'text_splitter', 'langchain')
    splitter_class = get_text_splitter_class(splitter_name)

    if splitter_name == "langchain":
        return splitter_class(
            chunk_size=getattr(settings, 'chunk_size', 1000),
            chunk_overlap=getattr(settings, 'chunk_overlap', 200)
        )
    elif splitter_name == "semantic":
        return splitter_class(
            chunk_size=getattr(settings, 'chunk_size', 1000),
            chunk_overlap=getattr(settings, 'chunk_overlap', 200)
        )
    else:
        raise ValueError(f"Unsupported text splitter: {splitter_name}")

def get_document_processor(
    settings: Settings = Depends(get_settings)
) -> DocumentPreProcessor:
    """Get the document processor with the configured components."""
    return DocumentPreProcessor()

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

def get_document_service(
    document_processor = Depends(get_document_processor),
    text_splitter = Depends(get_text_splitter),
    vector_store = Depends(get_vector_store),
    embedding_provider = Depends(get_embedding_provider)
) -> DocumentService:
    """Get the document service instance."""
    return DocumentService(
        document_processor=document_processor,
        text_splitter=text_splitter,
        vector_store=vector_store,
        embedding_provider=embedding_provider
    )

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
