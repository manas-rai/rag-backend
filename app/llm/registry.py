"""Registry for LLM providers."""

from app.llm import LLMProvider
from app.llm.providers.azure_provider import AzureLLMProvider
from app.llm.providers.gcp_provider import GCPLLMProvider
from app.llm.providers.aws_provider import AWSLLMProvider
from app.llm.providers.groq_provider import GroqLLMProvider

# Provider registry for easy access
PROVIDERS = {
    "azure": AzureLLMProvider,
    "gcp": GCPLLMProvider,
    "aws": AWSLLMProvider,
    "groq": GroqLLMProvider,
}

def get_provider_class(provider_name: str):
    """Get provider class by name."""
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")
    return PROVIDERS[provider_name]

def get_available_providers():
    """Get list of available provider names."""
    return list(PROVIDERS.keys())

# Export everything
__all__ = [
    "LLMProvider",
    "AzureLLMProvider", 
    "GCPLLMProvider",
    "AWSLLMProvider",
    "GroqLLMProvider",
    "PROVIDERS",
    "get_provider_class",
    "get_available_providers"
]
