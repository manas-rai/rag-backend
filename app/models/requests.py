from pydantic import BaseModel, Field
from typing import List, Optional
from .base import DocumentBase

class DocumentRequest(BaseModel):
    """Request model for document processing."""
    documents: List[DocumentBase] = Field(..., description="List of documents to process")

class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    query: str = Field(..., description="The query to process")
    max_chunks: int = Field(default=4, description="Maximum number of context chunks to retrieve")
    temperature: float = Field(default=0.7, description="Temperature for response generation")
    max_tokens: int = Field(default=500, description="Maximum tokens in the response") 