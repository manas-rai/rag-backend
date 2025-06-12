"""Models for requests in the RAG Backend."""

from typing import List, Optional
from fastapi import UploadFile
from pydantic import BaseModel, Field, field_validator
from app.models.base import DocumentBase, DocumentConfig, DocumentParams

class DocumentRequest(BaseModel):
    """Request model for document processing."""
    documents: List[DocumentBase] = Field(..., description="List of documents to process")
    config: Optional[DocumentConfig] = Field(
        default=None, description="Configuration for document processing")
    params: Optional[DocumentParams] = Field(
        default=None, description="Parameters for document processing")

class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    query: str = Field(..., description="The query to process")
    config: Optional[DocumentConfig] = Field(
        default=None, description="Configuration for document processing")
    params: Optional[DocumentParams] = Field(
        default=None, description="Parameters for document processing")

class PDFFile(BaseModel):
    """PDF file validation model."""
    file: UploadFile = Field(..., description="PDF file to upload")

    @field_validator('file')
    @classmethod
    def validate_file_type(cls, v: UploadFile) -> UploadFile:
        """Validate that the file is a PDF."""
        if not v.content_type == "application/pdf":
            raise ValueError("Only PDF files are allowed")
        return v

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
