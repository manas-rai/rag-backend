"""Registry for text splitters."""

from app.text import TextSplitter
from app.text.splitters.langchain_splitter import LangChainTextSplitter
from app.text.splitters.semantic_splitter import SemanticTextSplitter

# Provider registry for easy access
PROVIDERS = {
    "langchain": LangChainTextSplitter,
    "semantic": SemanticTextSplitter,
}

def get_splitter_class(splitter_name: str):
    """Get splitter class by name."""
    if splitter_name not in PROVIDERS:
        raise ValueError(f"Unsupported text splitter: {splitter_name}")
    return PROVIDERS[splitter_name]

def get_available_splitters():
    """Get list of available splitter names."""
    return list(PROVIDERS.keys())

# Export everything
__all__ = [
    "TextSplitter",
    "LangChainTextSplitter",
    "SemanticTextSplitter",
    "PROVIDERS",
    "get_splitter_class",
    "get_available_splitters"
]
