"""Custom exceptions for the RAG application."""

class RAGException(Exception):
    """Base exception for RAG application."""

class EmbeddingError(RAGException):
    """Exception raised for errors in embedding generation."""

class VectorStoreError(RAGException):
    """Exception raised for errors in vector store operations."""

class DocumentProcessingError(RAGException):
    """Exception raised for errors in document processing."""

class LLMError(RAGException):
    """Exception raised for errors in LLM operations."""

class ConfigurationError(RAGException):
    """Exception raised for configuration errors."""

class ValidationError(RAGException):
    """Exception raised for validation errors."""
