"""Service for document processing, chunking, and vector storage operations."""

from typing import Optional, List, Dict, Any
from app.document import DocumentPreProcessor
from app.text import TextSplitter
from app.vector import VectorStore
from app.embedding import EmbeddingProvider
from app.utils.logger import setup_logger

logger = setup_logger('document_service')

class DocumentService:
    """Service class for document processing, chunking, and vector storage operations."""

    def __init__(
        self,
        document_processor: DocumentPreProcessor,
        text_splitter: TextSplitter,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider
    ):
        """Initialize the document service with required components."""
        self.document_processor = document_processor
        self.text_splitter = text_splitter
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        logger.info("Document service initialized")

    async def process_and_store_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> str:
        """Process a document by chunking, creating embeddings, and storing in vector store."""
        try:
            # Update chunk parameters if provided
            if chunk_size is not None or chunk_overlap is not None:
                self.text_splitter.update_parameters(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

            # Split the document into chunks
            chunks = self.text_splitter.split_text(content)

            # Create embeddings for chunks
            embeddings = self.embedding_provider.get_embeddings(chunks)

            # store chunks with their embeddings
            doc_id = self.vector_store.add_documents(chunks, embeddings, metadata)
            return doc_id
        except Exception as e:
            logger.error("Failed to process and store document: %s", str(e))
            raise e

    async def process_and_store_batch(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[str]:
        """Process and store multiple documents in batch."""
        doc_ids = []

        # Update chunk parameters if provided
        if chunk_size is not None or chunk_overlap is not None:
            self.text_splitter.update_parameters(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        for doc in documents:
            # Split document into chunks
            chunks = self.text_splitter.split_text(doc["content"])

            # Create embeddings for chunks
            embeddings = await self.embedding_provider.get_embeddings(chunks)

            # Process and store chunks with their embeddings
            doc_id = await self.document_processor.process_document(
                chunks=chunks,
                embeddings=embeddings,
                metadata=doc.get("metadata")
            )
            doc_ids.append(doc_id)

        return doc_ids

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks from the vector store."""
        return await self.vector_store.delete_document(doc_id)

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document and its chunks from the vector store."""
        return await self.vector_store.get_document(doc_id)

    async def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List documents in the vector store."""
        return await self.vector_store.list_documents(limit, offset)

    async def process_pdf_and_store(self, file: bytes) -> str:
        """Process a PDF file and store it in the vector store."""
        try:
            # Process the document
            content = await self.document_processor.process_pdf_document(file)
            doc_id = await self.process_and_store_document(content)

            return doc_id
        except Exception as e:
            logger.error("Failed to process and store PDF: %s", str(e))
            raise e
