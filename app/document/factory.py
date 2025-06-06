"""Factory for creating document processors with different components."""

from typing import Optional
from app.document.processor import DocumentProcessor
from app.text.interfaces import TextSplitter, TextProcessor
from app.vector.interfaces import VectorStore
from app.text.splitters.langchain_splitter import LangChainTextSplitter
from app.vector.stores.chroma_store import ChromaVectorStore

class DocumentProcessorFactory:
    """Factory for creating document processors with different components."""

    @staticmethod
    def create_processor(
        text_splitter: TextSplitter,
        vector_store: VectorStore,
        text_processor: Optional[TextProcessor] = None
    ) -> DocumentProcessor:
        """Create a new document processor with the specified components."""
        return DocumentProcessor(
            text_splitter=text_splitter,
            vector_store=vector_store,
            text_processor=text_processor
        )

    @staticmethod
    def create_default_processor(
        vector_store_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_function = None
    ) -> DocumentProcessor:
        """Create a document processor with default components."""

        text_splitter = LangChainTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        vector_store = ChromaVectorStore(
            persist_directory=vector_store_path,
            embedding_function=embedding_function
        )

        return DocumentProcessor(
            text_splitter=text_splitter,
            vector_store=vector_store
        )
