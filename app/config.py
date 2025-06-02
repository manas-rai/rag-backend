from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, Literal, List

class Settings(BaseSettings):
    # LLM Provider settings
    llm_provider: Literal["openai", "gcp", "aws"] = "openai"

    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"

    # GCP settings (for future use)
    gcp_project_id: Optional[str] = None
    gcp_location: Optional[str] = None
    gcp_model: Optional[str] = None

    # AWS settings (for future use)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None
    aws_model: Optional[str] = None

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
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
