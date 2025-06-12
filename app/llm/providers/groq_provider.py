"""Groq implementation for LLM provider interface."""

from typing import List, Dict, Any
from groq import Groq
from app.llm import LLMProvider
from app.utils.logger import setup_logger

logger = setup_logger('groq_llm')

class GroqLLMProvider(LLMProvider):
    """Groq implementation of LLM provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "mixtral-8x7b-32768",
        embedding_model: str = "mixtral-8x7b-32768"
    ):
        """Initialize Groq client.
        
        Args:
            api_key: Groq API key
            model: Model name for text generation (default: mixtral-8x7b-32768)
            embedding_model: Model name for embeddings (default: mixtral-8x7b-32768)
        """
        self.client = Groq(api_key=api_key)
        self.model = model
        self.embedding_model = embedding_model
        logger.info("Initialized Groq client with model: %s", self.model)

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """Generate a response using Groq.
        
        Args:
            query: The user's query
            context: List of relevant context chunks
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Generated response text
        """
        # Create messages for the chat completion
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1000),
            temperature=kwargs.get("temperature", 0.7)
        )

        return response.choices[0].message.content

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Groq models being used."""
        return {
            "provider": "groq",
            "chat_model": self.model,
            "embedding_model": self.embedding_model
        }
