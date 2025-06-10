"""Main RAG service that orchestrates all components."""

from typing import Optional, List, Dict, Any
from app.llm import LLMProvider
from app.embedding import EmbeddingProvider
from app.vector import VectorStore
from app.text import TextSplitter
from app.document import DocumentPreProcessor

class RAGService:
    """Main service class that orchestrates all RAG components."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: Optional[EmbeddingProvider],
        vector_store: VectorStore,
        text_splitter: TextSplitter,
        document_processor: DocumentPreProcessor
    ):
        """Initialize the RAG service with all required components."""
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.text_splitter = text_splitter
        self.document_processor = document_processor
    
    async def process_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process a document and store it in the vector store."""
        # Split the document into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Process and store chunks
        doc_id = await self.document_processor.process_document(chunks, metadata)
        return doc_id
    
    async def query(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        # Get relevant documents from vector store
        relevant_docs = await self.vector_store.similarity_search(query_text, k=top_k)
        
        # Prepare context from relevant documents
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate response using LLM
        response = await self.llm_provider.generate(
            query=query_text,
            context=context
        )
        
        return {
            "response": response,
            "sources": [doc.metadata for doc in relevant_docs]
        }
    
    async def get_available_components(self) -> Dict[str, List[str]]:
        """Get information about available components."""
        return {
            "llm_provider": self.llm_provider.get_provider_info(),
            "embedding_provider": self.embedding_provider.get_provider_info() if self.embedding_provider else None,
            "vector_store": self.vector_store.get_provider_info(),
            "text_splitter": self.text_splitter.get_provider_info()
        } 