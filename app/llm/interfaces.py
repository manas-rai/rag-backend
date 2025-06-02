from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMProvider(ABC):
    """Interface for LLM providers."""
    
    @abstractmethod
    def generate_response(self, query: str, context: List[str], **kwargs) -> str:
        """Generate a response based on the query and context."""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for the given texts."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model being used."""
        pass 