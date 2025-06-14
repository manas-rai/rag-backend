"""Base models for the RAG Backend."""

from typing import Optional
from pydantic import BaseModel

class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = True
    error: Optional[str] = None
    message: Optional[str] = None

class DocumentBase(BaseModel):
    """Base document model."""
    content: str
    metadata: Optional[str] = None

class DocumentConfig(BaseModel):
    """Configuration for document processing."""
    llm_provider: Optional[str] = None
    embedding_provider: Optional[str] = None
    vector_store: Optional[str] = None
    chunker: Optional[str] = None

class DocumentParams(BaseModel):
    """Parameters for document processing."""
    max_chunks: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
