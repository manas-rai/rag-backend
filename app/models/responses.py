"""Models for responses in the RAG Backend."""

from typing import List
from pydantic import Field
from app.models.base import BaseResponse

class DocumentResponse(BaseResponse):
    """Response model for document processing."""
    message: str = Field(..., description="Status message")
    document_count: int = Field(..., description="Number of documents processed")

class QueryResponse(BaseResponse):
    """Response model for query results."""
    answer: str = Field(..., description="Generated answer based on context")
    sources: List[str] = Field(..., description="Relevant context chunks used for generation")
    model_used: str = Field(..., description="The LLM model used for generation")
