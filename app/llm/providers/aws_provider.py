from typing import List, Dict, Any
import boto3
from ..interfaces import LLMProvider
import json

class AWSLLMProvider(LLMProvider):
    """AWS implementation of LLM provider."""
    
    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        region: str,
        model: str,
        embedding_model: str
    ):
        """Initialize AWS clients.
        
        Args:
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            region: AWS region
            model: Model ID for text generation
            embedding_model: Model ID for embeddings
        """
        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region
        )
        self.model = model
        self.embedding_model = embedding_model
    
    def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        max_tokens: int = 1000
    ) -> str:
        """Generate a response using AWS Bedrock.
        
        Args:
            query: The user's query
            context: List of relevant context chunks
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Generated response text
        """
        # Format context into a single string
        context_text = "\n\n".join([chunk["text"] for chunk in context])
        
        # Create prompt
        prompt = f"""Context:
{context_text}

Question: {query}

Please provide a helpful answer based on the context above."""
        
        # Generate response using Claude
        response = self.bedrock.invoke_model(
            modelId=self.model,
            body=json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": max_tokens,
                "temperature": 0.7
            })
        )
        
        return json.loads(response["body"].read())["completion"]
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using AWS Bedrock.
        
        Args:
            texts: List of texts to get embeddings for
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            response = self.bedrock.invoke_model(
                modelId=self.embedding_model,
                body=json.dumps({
                    "inputText": text
                })
            )
            embeddings.append(json.loads(response["body"].read())["embedding"])
        return embeddings 