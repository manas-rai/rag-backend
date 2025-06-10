"""Constants used throughout the RAG application."""

# Vector Store Constants
VECTOR_STORE_TYPE_PINECONE = "pinecone"
VECTOR_STORE_METRIC_COSINE = "cosine"
VECTOR_STORE_METRIC_EUCLIDEAN = "euclidean"
VECTOR_STORE_METRIC_DOTPRODUCT = "dotproduct"
VECTOR_STORE_BATCH_SIZE = 100

# Embedding Provider Constants
EMBEDDING_PROVIDER_TYPE_SENTENCE_TRANSFORMER = "sentence_transformer"

# LLM Provider Constants
LLM_PROVIDER_TYPE_OPENAI = "openai"
LLM_PROVIDER_TYPE_AZURE = "azure"
LLM_PROVIDER_TYPE_GCP = "gcp"
LLM_PROVIDER_TYPE_AWS = "aws"
LLM_PROVIDER_TYPE_GROQ = "groq"

# Error Messages
ERROR_EMPTY_TEXT_LIST = "Empty text list provided"
ERROR_TEXTS_EMBEDDINGS_MISMATCH = "Number of texts must match number of embeddings"
ERROR_EMPTY_DOCUMENT = "Document is empty"
ERROR_INVALID_DOCUMENT_TYPE = "Invalid document type"
ERROR_INVALID_EMBEDDING_DIMENSION = "Embedding dimension must be {dimension}, got {actual}"
ERROR_NO_TEXTS_FOR_DELETION = "No texts provided for deletion"
ERROR_DOCUMENT_ID_REQUIRED = "Document ID is required"
ERROR_INVALID_PAGINATION = "Limit and offset must be non-negative"
ERROR_INVALID_METRIC = "Invalid metric type. Must be one of: cosine, euclidean, dotproduct"
ERROR_INVALID_PROVIDER = "Invalid provider type"
ERROR_INVALID_MODEL = "Invalid model name"
ERROR_INVALID_DEVICE = "Invalid device. Must be 'cpu' or 'cuda'"
ERROR_INVALID_API_KEY = "Invalid API key"
ERROR_INVALID_API_BASE = "Invalid API base URL"
ERROR_INVALID_API_VERSION = "Invalid API version"
ERROR_INVALID_DEPLOYMENT = "Invalid deployment name"
ERROR_INVALID_PROJECT = "Invalid project ID"
ERROR_INVALID_LOCATION = "Invalid location"
ERROR_INVALID_ACCESS_KEY = "Invalid access key"
ERROR_INVALID_SECRET_KEY = "Invalid secret key"
ERROR_INVALID_REGION = "Invalid region"
ERROR_NOT_FOUND = "Not found"

# Metadata Keys
METADATA_KEY_TEXT = "text"
METADATA_KEY_DOC_ID = "doc_id"
METADATA_KEY_CHUNK_INDEX = "chunk_index"
METADATA_KEY_TOTAL_CHUNKS = "total_chunks"
