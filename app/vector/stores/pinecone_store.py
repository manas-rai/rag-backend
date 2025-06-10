"""Pinecone implementation of vector store."""

import uuid
from typing import List, Dict, Any, Optional, Union
from pinecone import Pinecone
from app.vector import VectorStore
from utils.exceptions import VectorStoreError, ValidationError, ConfigurationError
from utils.constants import (
    VECTOR_STORE_TYPE_PINECONE,
    VECTOR_STORE_METRIC_COSINE,
    VECTOR_STORE_METRIC_EUCLIDEAN,
    VECTOR_STORE_METRIC_DOTPRODUCT,
    VECTOR_STORE_BATCH_SIZE,
    ERROR_INVALID_METRIC,
    ERROR_NO_TEXTS_FOR_DELETION,
    ERROR_DOCUMENT_ID_REQUIRED,
    ERROR_INVALID_PAGINATION,
    METADATA_KEY_TEXT,
    METADATA_KEY_DOC_ID,
    METADATA_KEY_CHUNK_INDEX,
    METADATA_KEY_TOTAL_CHUNKS
)
from utils.config import get_settings

class PineconeVectorStore(VectorStore):
    """Pinecone implementation of vector store."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: Optional[int] = None,
        metric: Optional[str] = None,
    ):
        """Initialize Pinecone vector store.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            dimension: Dimension of the embedding vectors
            metric: Distance metric for similarity search ('cosine', 'euclidean', 'dotproduct')
            
        Raises:
            ConfigurationError: If initialization fails
            ValidationError: If metric is invalid
        """
        settings = get_settings()
        self.dimension = dimension or settings.vector_store_dimension
        self.metric = metric or settings.vector_store_metric

        valid_metrics = [
            VECTOR_STORE_METRIC_COSINE,
            VECTOR_STORE_METRIC_EUCLIDEAN,
            VECTOR_STORE_METRIC_DOTPRODUCT
        ]
        if self.metric not in valid_metrics:
            raise ValidationError(ERROR_INVALID_METRIC)

        try:
            self.api_key = api_key
            self.index_name = index_name

            # Initialize Pinecone client
            self.pc = Pinecone(api_key=api_key)
            self._initialize_index()
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Pinecone vector store: {str(e)}") from e

    def _initialize_index(self) -> None:
        """Initialize or connect to the Pinecone index.
        
        Raises:
            ConfigurationError: If index initialization fails
        """
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                raise ConfigurationError(f"Index {self.index_name} does not exist")
            # Connect to the index
            self.index = self.pc.Index(self.index_name)

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Pinecone index: {str(e)}") from e

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    ) -> str:
        """Add texts to the Pinecone index.

        Args:
            texts: List of text chunks
            metadata: Optional metadata for the text(s)
        
        Returns:
            Document ID for the stored document
            
        Raises:
            ValidationError: If input validation fails
            VectorStoreError: If storage operation fails
        """
        try:
            # Generate a document ID
            doc_id = str(uuid.uuid4())

            # Prepare vectors for upsert
            vectors = []
            for i, text in enumerate(texts):
                chunk_id = f"{doc_id}_chunk_{i}"

                # Prepare metadata for this chunk
                chunk_metadata = {
                    METADATA_KEY_TEXT: text,
                    METADATA_KEY_DOC_ID: doc_id,
                    METADATA_KEY_CHUNK_INDEX: i,
                    METADATA_KEY_TOTAL_CHUNKS: len(texts)
                }

                # Add additional metadata if provided
                if metadatas:
                    if isinstance(metadatas, dict):
                        # Same metadata for all chunks
                        chunk_metadata.update(metadatas)
                    elif isinstance(metadatas, list) and i < len(metadatas):
                        # Different metadata for each chunk
                        chunk_metadata.update(metadatas[i])

                vectors.append({
                    "id": chunk_id,
                    "metadata": chunk_metadata
                })

            # Upsert vectors in batches
            for i in range(0, len(vectors), VECTOR_STORE_BATCH_SIZE):
                batch = vectors[i:i + VECTOR_STORE_BATCH_SIZE]
                self.index.upsert(vectors=batch)

            return doc_id

        except ValidationError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to add texts to Pinecone: {str(e)}") from e

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    ) -> str:
        """Add documents to the Pinecone index.
        
        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadata: Optional metadata for the document(s)
            
        Returns:
            Document ID for the stored document
        """
        if not texts or not embeddings:
            raise ValueError("Both texts and embeddings must be provided")

        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")

        # Generate a document ID
        doc_id = str(uuid.uuid4())

        # Prepare vectors for upsert
        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"

            # Prepare metadata for this chunk
            chunk_metadata = {
                "text": text,
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(texts)
            }

            # Add additional metadata if provided
            if metadata:
                if isinstance(metadata, dict):
                    # Same metadata for all chunks
                    chunk_metadata.update(metadata)
                elif isinstance(metadata, list) and i < len(metadata):
                    # Different metadata for each chunk
                    chunk_metadata.update(metadata[i])

            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": chunk_metadata
            })

        # Upsert vectors in batches (Pinecone recommends batch size of 100)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

        return doc_id

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar texts in the Pinecone index.
        
        Args:
            query: The query text
            k: Number of results to return
            filter: Optional filter criteria for the search
            
        Returns:
            List of dictionaries containing text and metadata for each result
            
        Raises:
            ValidationError: If input validation fails
            VectorStoreError: If search operation fails
        """
        try:
            # Perform similarity search
            search_results = self.index.query(
                vector=query,
                top_k=k,
                include_metadata=True,
                filter=filters
            )

            # Format results
            results = []
            for match in search_results.matches:
                result = {
                    METADATA_KEY_TEXT: match.metadata.get(METADATA_KEY_TEXT, ""),
                    "metadata": {
                        k: v for k, v in match.metadata.items()
                        if k not in [
                            METADATA_KEY_TEXT,
                            METADATA_KEY_DOC_ID,
                            METADATA_KEY_CHUNK_INDEX,
                            METADATA_KEY_TOTAL_CHUNKS
                        ]
                    },
                    "score": match.score
                }
                results.append(result)

            return results

        except ValidationError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to perform similarity search: {str(e)}") from e

    def delete(self, texts: List[str]) -> None:
        """Delete texts from the Pinecone index.
        
        Args:
            texts: List of texts to delete
            
        Raises:
            ValidationError: If input validation fails
            VectorStoreError: If deletion operation fails
        """
        try:
            if not texts:
                raise ValidationError(ERROR_NO_TEXTS_FOR_DELETION)

            # Delete all vectors with matching text in metadata
            self.index.delete(
                filter={METADATA_KEY_TEXT: {"$in": texts}}
            )
        except ValidationError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to delete texts: {str(e)}") from e

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document and all its chunks from the Pinecone index.
        
        Args:
            doc_id: ID of the document to retrieve
            
        Returns:
            Dictionary containing document chunks and metadata, or None if not found
            
        Raises:
            ValidationError: If input validation fails
            VectorStoreError: If retrieval operation fails
        """
        try:
            if not doc_id:
                raise ValidationError(ERROR_DOCUMENT_ID_REQUIRED)

            # Query to find all chunks of the document
            results = self.index.query(
                vector=[0.0] * self.dimension,  # Dummy vector of correct dimension
                top_k=1000,  # Get all chunks
                include_metadata=True,
                filter={METADATA_KEY_DOC_ID: doc_id}
            )

            if not results.matches:
                return None

            # Sort chunks by chunk_index
            chunks = sorted(
                results.matches,
                key=lambda x: x.metadata.get(METADATA_KEY_CHUNK_INDEX, 0)
            )

            # Reconstruct document
            document = {
                METADATA_KEY_DOC_ID: doc_id,
                "chunks": [match.metadata.get(METADATA_KEY_TEXT, "") for match in chunks],
                "metadata": {
                    k: v for k, v in chunks[0].metadata.items()
                    if k not in [
                        METADATA_KEY_TEXT,
                        METADATA_KEY_DOC_ID,
                        METADATA_KEY_CHUNK_INDEX,
                        METADATA_KEY_TOTAL_CHUNKS
                    ]
                }
            }

            return document

        except ValidationError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to retrieve document: {str(e)}") from e

    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List documents in the Pinecone index.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of document metadata
            
        Raises:
            ValidationError: If input validation fails
            VectorStoreError: If listing operation fails
        """
        try:
            if limit < 0 or offset < 0:
                raise ValidationError(ERROR_INVALID_PAGINATION)

            # Get all vectors with their metadata
            results = self.index.query(
                vector=[0.0] * self.dimension,  # Dummy vector of correct dimension
                top_k=1000,  # Get all documents
                include_metadata=True
            )

            # Group chunks by doc_id
            documents = {}
            for match in results.matches:
                doc_id = match.metadata.get(METADATA_KEY_DOC_ID)
                if doc_id:
                    if doc_id not in documents:
                        documents[doc_id] = {
                            METADATA_KEY_DOC_ID: doc_id,
                            "metadata": {
                                k: v for k, v in match.metadata.items()
                                if k not in [
                                    METADATA_KEY_TEXT,
                                    METADATA_KEY_DOC_ID,
                                    METADATA_KEY_CHUNK_INDEX,
                                    METADATA_KEY_TOTAL_CHUNKS
                                ]
                            },
                            METADATA_KEY_TOTAL_CHUNKS: match.metadata.get(
                                    METADATA_KEY_TOTAL_CHUNKS, 0
                                )
                        }

            # Convert to list and apply pagination
            doc_list = list(documents.values())
            return doc_list[offset:offset + limit]

        except ValidationError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to list documents: {str(e)}") from e

    def clear(self) -> None:
        """Clear all vectors from the Pinecone index.
        
        Raises:
            VectorStoreError: If clearing operation fails
        """
        try:
            self.index.delete(delete_all=True)
        except Exception as e:
            raise VectorStoreError(f"Failed to clear index: {str(e)}") from e

    def get_config(self) -> Dict[str, Any]:
        """Get information about the Pinecone provider.
        
        Returns:
            Dictionary containing provider information
            
        Raises:
            VectorStoreError: If config retrieval fails
        """
        try:
            return {
                "type": VECTOR_STORE_TYPE_PINECONE,
                "index_name": self.index_name,
                "dimension": self.dimension,
                "metric": self.metric
            }
        except Exception as e:
            raise VectorStoreError(f"Failed to get config: {str(e)}") from e
