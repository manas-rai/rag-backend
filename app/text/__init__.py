"""Text splitters and interfaces."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class TextSplitter(ABC):
    """Interface for text splitting strategies."""

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the splitter."""

# Import all text splitter providers
from app.text.splitters.langchain_splitter import LangChainTextSplitter
from app.text.splitters.semantic_splitter import SemanticTextSplitter

# Provider registry for easy access
SPLITTER_PROVIDERS = {
    "langchain": LangChainTextSplitter,
    "semantic": SemanticTextSplitter,
}

def get_splitter_class(provider_name: str):
    """Get text splitter provider class by name."""
    if provider_name not in SPLITTER_PROVIDERS:
        raise ValueError(f"Unsupported text splitter provider: {provider_name}")
    return SPLITTER_PROVIDERS[provider_name]

def get_available_splitters():
    """Get list of available text splitter provider names."""
    return list(SPLITTER_PROVIDERS.keys())

# Export everything
__all__ = [
    "TextSplitter",
    "LangChainTextSplitter",
    "SemanticTextSplitter",
    "SPLITTER_PROVIDERS",
    "get_splitter_class",
    "get_available_splitters"
]
