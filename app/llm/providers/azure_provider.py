"""Azure OpenAI implementation of LLM provider."""

from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from app.llm import LLMProvider
from utils.exceptions import LLMError, ConfigurationError, ValidationError
from utils.constants import (
    LLM_PROVIDER_TYPE_AZURE,
    ERROR_INVALID_API_KEY,
    ERROR_INVALID_API_BASE,
    ERROR_INVALID_API_VERSION,
    ERROR_INVALID_DEPLOYMENT
)
from utils.config import get_settings

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
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Azure OpenAI client: {str(e)}") from e

    def generate(
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
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"Failed to generate text: {str(e)}") from e

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the Azure OpenAI provider.
        
        Returns:
            Dictionary containing provider information
            
        Raises:
            LLMError: If info retrieval fails
        """
        try:
            return {
                "type": LLM_PROVIDER_TYPE_AZURE,
                "deployment": self.deployment_name
            }
        except Exception as e:
            raise LLMError(f"Failed to get provider info: {str(e)}") from e

    def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate a response using Azure OpenAI.
        
        Args:
            query: The user's query
            context: List of relevant context chunks
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Generated response text
        """
        # Format context into a single string
        context_text = "\n\n".join([chunk["text"] for chunk in context])

        # Create messages for the chat completion
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers based on context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {query}"
            }
        ]

        # Generate response
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )

        return response.choices[0].message.content

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using Azure OpenAI.
        
        Args:
            texts: List of texts to get embeddings for
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model=self.deployment_name,
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Azure OpenAI models being used."""
        return {
            "provider": "azure",
            "chat_model": self.deployment_name
        }
