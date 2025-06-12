"""Service for query processing operations."""

from typing import Optional, List, Dict, Any
from app.llm import LLMProvider
from app.vector import VectorStore
from app.embedding import EmbeddingProvider
from app.document import DocumentPreProcessor
from app.prompts.prompt_manager import PromptManager
from app.utils.logger import setup_logger

logger = setup_logger('query_service')

class QueryService:
    """Service class for query processing operations."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        document_processor: DocumentPreProcessor
    ):
        """Initialize the query service with required components."""
        self.llm_provider = llm_provider
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.document_processor = document_processor
        self.prompt_manager = PromptManager()

    async def query_vector(
        self,
        query_text: str,
        top_k: int = 3,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        try:
            # Convert query text into embeddings
            query_embedding = self.embedding_provider.get_embeddings([query_text])

            # Get relevant documents from vector store
            relevant_docs = self.vector_store.similarity_search(
                query=query_text,
                query_embedding=query_embedding,
                k=top_k
            )

            # Extract content from relevant documents
            context = "\n\n".join([doc.get("text") for doc in relevant_docs])

            # Get system prompt to set behavior
            system_prompt = self.prompt_manager.format_prompt(
                prompt_type="system",
                prompt_name="rag_system_prompt",
                context="",  # System prompt doesn't need context
                query=""     # System prompt doesn't need query
            )

            # Get user prompt with specific task and context
            user_prompt = self.prompt_manager.format_prompt(
                prompt_type="user",
                prompt_name="rag_user_prompt",
                context=context,
                query=query_text,
                temperature=temperature or 0.7,
                max_tokens=max_tokens or 500,
                num_sources=len(relevant_docs),
                chunk_size=len(context.split()),  # Approximate chunk size
                similarity_threshold=0.7,  # Default similarity threshold
                query_type="general",  # Can be customized based on query analysis
                document_types=[doc.get("metadata") for doc in relevant_docs],
                time_period="current"  # Can be extracted from metadata if available
            )

            # Generate response using LLM with both prompts
            response = self.llm_provider.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return {
                "response": response,
                "sources": [doc.get("text") for doc in relevant_docs]
            }
        except Exception as e:
            logger.error("Error querying RAG system: %s", str(e))
            raise

    async def get_similar_documents(
        self,
        query_text: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Get similar documents without generating a response."""
        try:
            query_embedding = self.embedding_provider.get_embeddings([query_text])
            relevant_docs = self.vector_store.similarity_search(
                query=query_text,
                query_embedding=query_embedding,
                k=top_k
            )
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in relevant_docs]
        except Exception as e:
            logger.error("Error getting similar documents: %s", str(e))
            raise
