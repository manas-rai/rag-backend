from typing import List, Dict, Any
from openai import AzureOpenAI
from ..interfaces import LLMProvider

class AzureLLMProvider(LLMProvider):
    """Azure OpenAI implementation of LLM provider."""
    
    def __init__(
        self,
        api_key: str,
        api_base: str,
        api_version: str,
        deployment_name: str,
        embedding_deployment_name: str
    ):
        """Initialize Azure OpenAI client.
        
        Args:
            api_key: Azure OpenAI API key
            api_base: Azure OpenAI API base URL
            api_version: Azure OpenAI API version
            deployment_name: Azure OpenAI deployment name for chat completions
            embedding_deployment_name: Azure OpenAI deployment name for embeddings
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_base
        )
        self.deployment_name = deployment_name
        self.embedding_deployment_name = embedding_deployment_name
    
    def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        max_tokens: int = 1000
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
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
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
                model=self.embedding_deployment_name,
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings 