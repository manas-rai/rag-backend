from typing import Dict, Type
from .interfaces import LLMProvider
from .providers.openai_provider import OpenAIProvider

class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    _providers: Dict[str, Type[LLMProvider]] = {
        "openai": OpenAIProvider,
        # Add other providers here as they are implemented
        # "gcp": GCPProvider,
        # "aws": AWSBedrockProvider,
    }
    
    @classmethod
    def create_provider(cls, provider_type: str, **kwargs) -> LLMProvider:
        """Create a new LLM provider instance."""
        if provider_type not in cls._providers:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        provider_class = cls._providers[provider_type]
        return provider_class(**kwargs)
    
    @classmethod
    def register_provider(cls, provider_type: str, provider_class: Type[LLMProvider]):
        """Register a new provider type."""
        cls._providers[provider_type] = provider_class 