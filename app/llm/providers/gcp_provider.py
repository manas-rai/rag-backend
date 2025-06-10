"""Google Cloud implementation of LLM provider."""

from typing import Dict, Any, Optional
from vertexai import generative_models
from app.llm import LLMProvider
from utils.exceptions import LLMError, ConfigurationError, ValidationError
from utils.constants import (
    LLM_PROVIDER_TYPE_GCP,
    ERROR_INVALID_PROJECT,
    ERROR_INVALID_LOCATION,
    ERROR_INVALID_MODEL
)
from utils.config import get_settings

class GCPLLMProvider(LLMProvider):
    """Google Cloud implementation of LLM provider."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model: Optional[str] = None
    ):
        """Initialize Google Cloud provider.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud location
            model: Model name
            
        Raises:
            ConfigurationError: If initialization fails
            ValidationError: If input validation fails
        """
        settings = get_settings()

        if not project_id and not settings.gcp_project_id:
            raise ValidationError(ERROR_INVALID_PROJECT)
        if not location and not settings.gcp_location:
            raise ValidationError(ERROR_INVALID_LOCATION)
        if not model and not settings.gcp_model:
            raise ValidationError(ERROR_INVALID_MODEL)

        try:
            self.project_id = project_id or settings.gcp_project_id
            self.location = location or settings.gcp_location
            self.model = model or settings.gcp_model

            # Initialize Vertex AI
            generative_models.init(
                project=self.project_id,
                location=self.location
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Google Cloud client: {str(e)}") from e

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using Google Cloud.
        
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
            model = generative_models.GenerativeModel(self.model)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature or settings.llm_temperature,
                    "max_output_tokens": max_tokens or settings.llm_max_tokens
                }
            )
            return response.text
        except Exception as e:
            raise LLMError(f"Failed to generate text: {str(e)}") from e

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the Google Cloud provider.
        
        Returns:
            Dictionary containing provider information
            
        Raises:
            LLMError: If info retrieval fails
        """
        try:
            return {
                "type": LLM_PROVIDER_TYPE_GCP,
                "model": self.model,
                "project": self.project_id,
                "location": self.location
            }
        except Exception as e:
            raise LLMError(f"Failed to get provider info: {str(e)}") from e
