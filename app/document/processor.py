from typing import List, Dict, Any, Optional
from ..text.interfaces import TextSplitter, TextProcessor
from ..vector.interfaces import VectorStore
from ..llm.interfaces import LLMProvider

class DocumentProcessor:
    """Document processor that coordinates text splitting, processing, and storage."""
    
    def __init__(
        self,
        text_splitter: TextSplitter,
        vector_store: VectorStore,
        text_processor: Optional[TextProcessor] = None
    ):
        self.text_splitter = text_splitter
        self.vector_store = vector_store
        self.text_processor = text_processor
    
    def process_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Process and store documents."""
        if not documents:
            return
            
        # Process texts if processor is available
        if self.text_processor:
            documents = [self.text_processor.process_text(doc) for doc in documents]
        
        # Split documents into chunks
        all_chunks = []
        all_metadatas = []
        
        for i, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc)
            all_chunks.extend(chunks)
            
            # If metadatas are provided, duplicate them for each chunk
            if metadatas:
                doc_metadata = metadatas[i] if i < len(metadatas) else {}
                all_metadatas.extend([doc_metadata] * len(chunks))
        
        # Store chunks in vector store
        self.vector_store.add_texts(all_chunks, all_metadatas)
    
    def get_relevant_chunks(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query."""
        return self.vector_store.similarity_search(query, k=k)
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the document processor."""
        return {
            "text_splitter": self.text_splitter.get_config(),
            "vector_store": self.vector_store.get_config(),
            "has_text_processor": self.text_processor is not None
        } 