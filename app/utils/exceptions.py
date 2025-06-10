"""Custom exceptions for the RAG application."""

class RAGException(Exception):
    """Base exception for RAG application."""
    pass

class EmbeddingError(RAGException):
    """Exception raised for errors in embedding generation."""
    pass

class VectorStoreError(RAGException):
    """Exception raised for errors in vector store operations."""
    pass

class DocumentProcessingError(RAGException):
    """Exception raised for errors in document processing."""
    pass

class LLMError(RAGException):
    """Exception raised for errors in LLM operations."""
    pass

class ConfigurationError(RAGException):
    """Exception raised for configuration errors."""
    pass

class ValidationError(RAGException):
    """Exception raised for validation errors."""
    pass 