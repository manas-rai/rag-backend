"""Pinecone implementation of vector store."""

import uuid
from typing import List, Dict, Any, Optional, Union
from pinecone import Pinecone
from app.vector import VectorStore
from app.exceptions import VectorStoreError, ValidationError, ConfigurationError

class PineconeVectorStore(VectorStore):
    """Pinecone implementation of vector store."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: Optional[int] = 768,
        metric: Optional[str] = "cosine",
    ):
        """Initialize Pinecone vector store.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            dimension: Dimension of the embedding vectors
            metric: Distance metric for similarity search ('cosine', 'euclidean', 'dotproduct')
            
        Raises:
            ConfigurationError: If initialization fails
        """
        try:
            self.api_key = api_key
            self.index_name = index_name
            self.dimension = dimension
            self.metric = metric

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
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                # Create new index with serverless spec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-west-2"
                        }
                    }
                )

            # Connect to the index
            self.index = self.pc.Index(self.index_name)

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Pinecone index: {str(e)}") from e

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    ) -> str:
        """Add texts to the Pinecone index.

        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadata: Optional metadata for the text(s)
        
        Returns:
            Document ID for the stored document
            
        Raises:
            ValidationError: If input validation fails
            VectorStoreError: If storage operation fails
        """
        try:
            if not texts or not embeddings:
                raise ValidationError("Both texts and embeddings must be provided")
            
            if len(texts) != len(embeddings):
                raise ValidationError("Number of texts must match number of embeddings")
                
            # Validate embedding dimensions
            for embedding in embeddings:
                if len(embedding) != self.dimension:
                    raise ValidationError(f"Embedding dimension must be {self.dimension}, got {len(embedding)}")

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
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar texts in the Pinecone index.
        
        Args:
            query_embedding: The query embedding vector
            k: Number of results to return
            filter: Optional filter criteria for the search
            
        Returns:
            List of dictionaries containing text and metadata for each result
            
        Raises:
            ValidationError: If input validation fails
            VectorStoreError: If search operation fails
        """
        try:
            # Validate query embedding dimension
            if len(query_embedding) != self.dimension:
                raise ValidationError(f"Query embedding dimension must be {self.dimension}, got {len(query_embedding)}")

            # Perform similarity search
            search_results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                filter=filter
            )

            # Format results
            results = []
            for match in search_results.matches:
                result = {
                    "text": match.metadata.get("text", ""),
                    "metadata": {k: v for k, v in match.metadata.items() if k != "text"},
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
                raise ValidationError("No texts provided for deletion")
                
            # Delete all vectors with matching text in metadata
            self.index.delete(
                filter={"text": {"$in": texts}}
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
                raise ValidationError("Document ID is required")
                
            # Query to find all chunks of the document
            results = self.index.query(
                vector=[0.0] * self.dimension,  # Dummy vector of correct dimension
                top_k=1000,  # Get all chunks
                include_metadata=True,
                filter={"doc_id": doc_id}
            )

            if not results.matches:
                return None

            # Sort chunks by chunk_index
            chunks = sorted(
                results.matches,
                key=lambda x: x.metadata.get("chunk_index", 0)
            )

            # Reconstruct document
            document = {
                "doc_id": doc_id,
                "chunks": [match.metadata.get("text", "") for match in chunks],
                "metadata": {k: v for k, v in chunks[0].metadata.items() 
                           if k not in ["text", "doc_id", "chunk_index", "total_chunks"]}
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
                raise ValidationError("Limit and offset must be non-negative")
                
            # Get all vectors with their metadata
            results = self.index.query(
                vector=[0.0] * self.dimension,  # Dummy vector of correct dimension
                top_k=1000,  # Get all documents
                include_metadata=True
            )

            # Group chunks by doc_id
            documents = {}
            for match in results.matches:
                doc_id = match.metadata.get("doc_id")
                if doc_id:
                    if doc_id not in documents:
                        documents[doc_id] = {
                            "doc_id": doc_id,
                            "metadata": {k: v for k, v in match.metadata.items() 
                                       if k not in ["text", "doc_id", "chunk_index", "total_chunks"]},
                            "total_chunks": match.metadata.get("total_chunks", 0)
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
                "type": "pinecone",
                "index_name": self.index_name,
                "dimension": self.dimension,
                "metric": self.metric
            }
        except Exception as e:
            raise VectorStoreError(f"Failed to get config: {str(e)}") from e
