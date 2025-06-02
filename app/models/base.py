"""Base models for the RAG Backend."""

from typing import Optional
from pydantic import BaseModel

class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = True
    error: Optional[str] = None

class DocumentBase(BaseModel):
    """Base document model."""
    content: str
    metadata: Optional[str] = None
