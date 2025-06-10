"""Configuration settings for the RAG Backend."""

import os
from functools import lru_cache
from typing import Optional, Literal
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Configuration settings for the RAG Backend."""

    model_config = ConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore"  # Ignore extra environment variables
    )

    # LLM Provider settings
    llm_provider: Literal[
        "azure", "gcp", "aws", "groq", "sentence_transformers"
    ] = os.getenv("LLM_PROVIDER", "groq")
    embedding_provider: Literal[
        "azure", "gcp", "aws", "groq", "sentence_transformers"
    ] = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")

    # Vector store settings
    vector_provider: Literal["chroma", "pinecone"] = os.getenv("VECTOR_PROVIDER", "pinecone")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    vector_store_dimension: int = int(os.getenv("VECTOR_STORE_DIMENSION", "768"))
    vector_store_metric: str = os.getenv("VECTOR_STORE_METRIC", "cosine")

    # Text processing settings
    text_splitter: Literal["langchain", "semantic"] = os.getenv("TEXT_SPLITTER", "langchain")

    # Azure settings
    azure_api_key: Optional[str] = os.getenv("AZURE_API_KEY")
    azure_api_base: Optional[str] = os.getenv("AZURE_API_BASE")
    azure_api_version: str = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
    azure_deployment_name: str = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-35-turbo")
    azure_embedding_deployment_name: str = os.getenv(
        "AZURE_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002"
    )

    # OpenAI settings
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

    # GCP settings
    gcp_project_id: Optional[str] = os.getenv("GCP_PROJECT_ID")
    gcp_location: Optional[str] = os.getenv("GCP_LOCATION")
    gcp_model: str = os.getenv("GCP_MODEL", "gemini-pro")
    gcp_embedding_model: str = os.getenv("GCP_EMBEDDING_MODEL", "textembedding-gecko")

    # AWS settings
    aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region: Optional[str] = os.getenv("AWS_REGION")
    aws_model: str = os.getenv("AWS_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")
    aws_embedding_model: str = os.getenv("AWS_EMBEDDING_MODEL", "amazon.titan-embed-text-v1")

    # Groq settings
    groq_api_key: str = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    groq_embedding_model: str = os.getenv("GROQ_EMBEDDING_MODEL", "llama3-8b-8192")

    # Sentence Transformers settings
    sentence_transformer_model: str = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    sentence_transformer_device: str = os.getenv("SENTENCE_TRANSFORMER_DEVICE", "cpu")

    # RAG settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Pinecone settings
    pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "ragindex1")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")

@lru_cache()
def get_settings() -> Settings:
    """Get the settings instance."""
    return Settings()
