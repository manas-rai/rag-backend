"""Azure OpenAI implementation of LLM provider."""

from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from app.llm import LLMProvider
from app.utils.exceptions import LLMError, ConfigurationError, ValidationError
from app.utils.constants import (
    LLM_PROVIDER_TYPE_AZURE,
    ERROR_INVALID_API_KEY,
    ERROR_INVALID_API_BASE,
    ERROR_INVALID_API_VERSION,
    ERROR_INVALID_DEPLOYMENT
)
from app.utils.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger('azure_llm')

class AzureLLMProvider(LLMProvider):
    """Azure OpenAI implementation of LLM provider."""

    def __init__(
        self,
        api_key: str,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None
    ):
        """Initialize Azure OpenAI provider.
        
        Args:
            api_key: Azure OpenAI API key
            api_base: Azure OpenAI API base URL
            api_version: Azure OpenAI API version
            deployment_name: Azure OpenAI deployment name
            
        Raises:
            ConfigurationError: If initialization fails
            ValidationError: If input validation fails
        """
        settings = get_settings()

        if not api_key and not settings.azure_api_key:
            raise ValidationError(ERROR_INVALID_API_KEY)
        if not api_base and not settings.azure_api_base:
            raise ValidationError(ERROR_INVALID_API_BASE)
        if not api_version and not settings.azure_api_version:
            raise ValidationError(ERROR_INVALID_API_VERSION)
        if not deployment_name and not settings.azure_deployment_name:
            raise ValidationError(ERROR_INVALID_DEPLOYMENT)

        try:
            self.client = AzureOpenAI(
                api_key=api_key or settings.azure_api_key,
                api_version=api_version or settings.azure_api_version,
                azure_endpoint=api_base or settings.azure_api_base
            )
            self.deployment_name = deployment_name or settings.azure_deployment_name
            logger.info("Initialized Azure OpenAI provider with deployment: %s",
                        self.deployment_name)
        except Exception as e:
            logger.error("Failed to initialize Azure OpenAI client: %s", str(e))
            raise ConfigurationError(f"Failed to initialize Azure OpenAI client: {str(e)}") from e

    def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using Azure OpenAI.
        
        Args:
            prompt: The prompt to generate text from
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
            
        Raises:
            LLMError: If generation fails
        """
        try:
            settings = get_settings()
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature or settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens
            )
            logger.info("Generated text: %s", response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Failed to generate text: %s", str(e))
            raise LLMError(f"Failed to generate text: {str(e)}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Azure OpenAI models being used."""
        return {
            "provider": "azure",
            "chat_model": self.deployment_name
        }
