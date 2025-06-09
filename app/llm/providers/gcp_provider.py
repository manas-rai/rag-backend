"""Google Cloud Platform implementation for LLM provider interface."""

from typing import List, Dict, Any
import google.generativeai as genai
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from app.llm import LLMProvider

class GCPLLMProvider(LLMProvider):
    """Google Cloud Platform implementation of LLM provider."""

    def __init__(
        self,
        project_id: str,
        location: str,
        model: str,
        embedding_model: str
    ):
        """Initialize GCP clients.
        
        Args:
            project_id: GCP project ID
            location: GCP location
            model: Model name for text generation
            embedding_model: Model name for embeddings
        """
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)

        # Initialize Gemini
        genai.configure()

        self.model = genai.GenerativeModel(model)
        self.embedding_model = embedding_model

    def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate a response using GCP's Gemini model.
        
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

        # Generate response
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7
            )
        )

        return response.text

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Vertex AI.
        
        Args:
            texts: List of texts to get embeddings for
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        model = TextEmbeddingModel.from_pretrained(self.embedding_model)
        for text in texts:
            response = model.get_embeddings([text])
            embeddings.append(response[0].values)
        return embeddings

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the GCP models being used."""
        return {
            "provider": "gcp",
            "chat_model": self.model.model_name,
            "embedding_model": self.embedding_model
        }
