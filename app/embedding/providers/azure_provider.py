"""Azure OpenAI implementation for embedding provider interface."""

from typing import List, Dict, Any
from openai import AzureOpenAI
from app.embedding import EmbeddingProvider
from app.utils.logger import setup_logger

logger = setup_logger('azure_embedding')

class AzureEmbeddingProvider(EmbeddingProvider):
    """Azure OpenAI implementation of embedding provider."""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        api_version: str,
        embedding_deployment_name: str
    ):
        """Initialize Azure OpenAI client for embeddings.
        
        Args:
            api_key: Azure OpenAI API key
            api_base: Azure OpenAI API base URL
            api_version: Azure OpenAI API version
            embedding_deployment_name: Azure OpenAI deployment name for embeddings
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_base
        )
        self.embedding_deployment_name = embedding_deployment_name
        logger.info("Initialized Azure OpenAI embedding provider with deployment: %s",
                    self.embedding_deployment_name)

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
                model=self.embedding_deployment_name,
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            The dimension of the embedding vectors (1536 for text-embedding-ada-002)
        """
        # Most Azure OpenAI embedding models use 1536 dimensions
        return 1536

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Azure OpenAI embedding model being used."""
        return {
            "provider": "azure",
            "embedding_model": self.embedding_deployment_name,
            "dimension": self.get_embedding_dimension()
        }
