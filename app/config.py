"""Configuration settings for the RAG Backend."""

from functools import lru_cache
from typing import Optional, Literal, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configuration settings for the RAG Backend."""

    # LLM Provider settings
    llm_provider: Literal["azure", "gcp", "aws"] = "azure"

    # Azure settings
    azure_api_key: Optional[str] = None
    azure_api_base: Optional[str] = None
    azure_api_version: str = "2024-02-15-preview"
    azure_deployment_name: str = "gpt-35-turbo"
    azure_embedding_deployment_name: str = "text-embedding-ada-002"

    # GCP settings
    gcp_project_id: Optional[str] = None
    gcp_location: Optional[str] = None
    gcp_model: str = "gemini-pro"
    gcp_embedding_model: str = "textembedding-gecko"

    # AWS settings
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None
    aws_model: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    aws_embedding_model: str = "amazon.titan-embed-text-v1"

    # Vector store settings
    vector_store_path: str = "./data/vector_store"

    # RAG settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # CORS settings
    cors_origins: List[str] = ["*"]  # Default to allow all origins
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]  # Default to allow all methods
    cors_allow_headers: List[str] = ["*"]  # Default to allow all headers

    class Config:
        """Pydantic configuration settings."""
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    """Get the settings instance."""
    return Settings()
