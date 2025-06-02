from typing import List, Dict, Any
from openai import OpenAI
from ..interfaces import LLMProvider

class OpenAIProvider(LLMProvider):
    """OpenAI implementation of the LLM provider interface."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_response(self, query: str, context: List[str], **kwargs) -> str:
        """Generate response using OpenAI's chat completion."""
        context_str = "\n\n".join(context)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. If the answer cannot be found in the context, say so."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 500)
        )
        
        return response.choices[0].message.content
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI's embedding model."""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [data.embedding for data in response.data]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OpenAI model being used."""
        return {
            "provider": "openai",
            "model": self.model,
            "embedding_model": "text-embedding-ada-002"
        } 