"""AWS Bedrock implementation of LLM provider."""

import json
from typing import Dict, Any, Optional
import boto3
from app.llm import LLMProvider
from app.utils.exceptions import LLMError, ConfigurationError, ValidationError
from app.utils.constants import (
    ERROR_INVALID_ACCESS_KEY,
    ERROR_INVALID_SECRET_KEY,
    ERROR_INVALID_REGION,
    ERROR_INVALID_MODEL
)
from app.utils.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger('aws_llm')

class AWSLLMProvider(LLMProvider):
    """AWS Bedrock implementation of LLM provider."""

    def __init__(
        self,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        region: Optional[str] = None,
        model: Optional[str] = None
    ):
        """Initialize AWS Bedrock provider.
        
        Args:
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            region: AWS region
            model: AWS Bedrock model ID
            
        Raises:
            ConfigurationError: If initialization fails
            ValidationError: If input validation fails
        """
        settings = get_settings()

        if not access_key_id and not settings.aws_access_key_id:
            raise ValidationError(ERROR_INVALID_ACCESS_KEY)
        if not secret_access_key and not settings.aws_secret_access_key:
            raise ValidationError(ERROR_INVALID_SECRET_KEY)
        if not region and not settings.aws_region:
            raise ValidationError(ERROR_INVALID_REGION)
        if not model and not settings.aws_model:
            raise ValidationError(ERROR_INVALID_MODEL)

        try:
            self.client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=access_key_id or settings.aws_access_key_id,
                aws_secret_access_key=secret_access_key or settings.aws_secret_access_key,
                region_name=region or settings.aws_region
            )
            self.model = model or settings.aws_model
            logger.info("Initialized AWS Bedrock provider with model: %s", self.model)
        except Exception as e:
            logger.error("Failed to initialize AWS Bedrock client: %s", str(e))
            raise ConfigurationError(f"Failed to initialize AWS Bedrock client: {str(e)}") from e

    def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using AWS Bedrock.
        
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
            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps({
                    "prompt": prompt,
                    "temperature": temperature or settings.llm_temperature,
                    "max_tokens": max_tokens or settings.llm_max_tokens
                })
            )
            logger.info("Generated text: %s", json.loads(response['body'].read())['completion'])
            return json.loads(response['body'].read())['completion']
        except Exception as e:
            logger.error("Failed to generate text: %s", str(e))
            raise LLMError(f"Failed to generate text: {str(e)}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the AWS models being used."""
        logger.info("Retrieving model info for AWS Bedrock")
        return {
            "provider": "aws",
            "chat_model": self.model,
            "embedding_model": self.model
        }
