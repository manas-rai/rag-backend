"""LLM providers and interfaces."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMProvider(ABC):
    """Interface for LLM providers."""

    @abstractmethod
    def generate_response(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate a response based on the query and context."""

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model being used."""
