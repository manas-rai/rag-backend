"""Service for query processing operations."""

from typing import Optional, List, Dict, Any
from app.llm import LLMProvider
from app.vector import VectorStore
from app.embedding import EmbeddingProvider

class QueryService:
    """Service class for query processing operations."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        vector_store: VectorStore,
        embedding_provider: Optional[EmbeddingProvider] = None
    ):
        """Initialize the query service with required components."""
        self.llm_provider = llm_provider
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    async def query(
        self,
        query_text: str,
        top_k: int = 3,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        # Get relevant documents from vector store
        relevant_docs = await self.vector_store.similarity_search(query_text, k=top_k)

        # Prepare context from relevant documents
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # Generate response using LLM with optional parameters
        response = await self.llm_provider.generate(
            query=query_text,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            "response": response,
            "sources": [doc.metadata for doc in relevant_docs]
        }

    async def get_similar_documents(
        self,
        query_text: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Get similar documents without generating a response."""
        relevant_docs = await self.vector_store.similarity_search(query_text, k=top_k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in relevant_docs]
