"""Factory for creating LLM providers."""

from typing import Type
from app.llm.interfaces import LLMProvider
from app.llm.providers.azure_provider import AzureLLMProvider
from app.llm.providers.gcp_provider import GCPLLMProvider
from app.llm.providers.aws_provider import AWSLLMProvider

class LLMProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> LLMProvider:
        """Create an LLM provider instance.
        
        Args:
            provider_type: Type of provider to create ("azure", "gcp", or "aws")
            **kwargs: Provider-specific configuration
            
        Returns:
            LLMProvider instance
            
        Raises:
            ValueError: If provider_type is not supported
        """
        if provider_type == "azure":
            return AzureLLMProvider(
                api_key=kwargs["api_key"],
                api_base=kwargs["api_base"],
                api_version=kwargs["api_version"],
                deployment_name=kwargs["deployment_name"],
                embedding_deployment_name=kwargs["embedding_deployment_name"]
            )
        elif provider_type == "gcp":
            return GCPLLMProvider(
                project_id=kwargs["project_id"],
                location=kwargs["location"],
                model=kwargs["model"],
                embedding_model=kwargs["embedding_model"]
            )
        elif provider_type == "aws":
            return AWSLLMProvider(
                access_key_id=kwargs["access_key_id"],
                secret_access_key=kwargs["secret_access_key"],
                region=kwargs["region"],
                model=kwargs["model"],
                embedding_model=kwargs["embedding_model"]
            )
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")

    @classmethod
    def register_provider(cls, provider_type: str, provider_class: Type[LLMProvider]) -> None:
        """Register a new provider type."""
        # This method is not used in the new implementation
