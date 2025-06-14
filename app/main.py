"""FastAPI application entry point for RAG Backend."""

from typing import Dict, Any
import os
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.routers import documents, query
from app.llm.registry import get_available_providers as get_llm_providers
from app.embedding.registry import get_available_providers as get_embedding_providers
from app.vector.registry import get_available_providers as get_vector_providers
from app.text.registry import get_available_splitters as get_text_splitters
from app.utils.config import get_settings
from app.utils.logger import setup_logger

# Set up logger
logger = setup_logger('main')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    logger.info("Starting up RAG Backend application...")
    settings = get_settings()
    logger.info("Using LLM provider: %s", settings.llm_provider)
    logger.info("Using embedding provider: %s", settings.embedding_provider)
    logger.info("Using vector provider: %s", getattr(settings, 'vector_provider', 'chroma'))
    logger.info("Using text splitter: %s", getattr(settings, 'text_splitter', 'langchain'))
    yield
    # Shutdown
    logger.info("Shutting down RAG Backend application...")

app = FastAPI(
    title="RAG Backend",
    description="Retrieval-Augmented Generation Backend API",
    version="1.0.0",
    lifespan=lifespan
)

# Parse CORS settings from environment variables
def parse_cors_list(env_var: str, default: list) -> list:
    """Parse comma-separated environment variable into list."""
    value = os.getenv(env_var)
    if value:
        return [item.strip() for item in value.split(",")]
    return default

def parse_cors_bool(env_var: str, default: bool) -> bool:
    """Parse boolean environment variable."""
    value = os.getenv(env_var)
    if value:
        return value.lower() == "true"
    return default

# Configure CORS
cors_origins = parse_cors_list("CORS_ORIGINS", ["*"])
cors_allow_methods = parse_cors_list("CORS_ALLOW_METHODS", ["*"])
cors_allow_headers = parse_cors_list("CORS_ALLOW_HEADERS", ["*"])
cors_allow_credentials = parse_cors_bool("CORS_ALLOW_CREDENTIALS", True)

logger.info("Configuring CORS with origins: %s", cors_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=cors_allow_methods,
    allow_headers=cors_allow_headers,
)

# Include routers
app.include_router(documents.router)
app.include_router(query.router)
logger.info("Routers configured successfully")

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with component information."""
    logger.info("Root endpoint accessed")
    settings = get_settings()
    logger.debug("Retrieved application settings")

    return {
        "message": "Welcome to RAG Backend API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "default_configuration": {
            "llm_provider": settings.llm_provider,
            "embedding_provider": settings.embedding_provider,
            "vector_provider": getattr(settings, 'vector_provider', 'chroma'),
            "text_splitter": getattr(settings, 'text_splitter', 'langchain')
        },
        "available_components": {
            "llm_providers": get_llm_providers(),
            "embedding_providers": get_embedding_providers(),
            "vector_providers": get_vector_providers(),
            "text_splitters": get_text_splitters()
        },
        "configuration_info": {
            "description": "Configure components via environment variables",
            "environment_variables": {
                "LLM_PROVIDER": "Choose from: " + ", ".join(get_llm_providers()),
                "EMBEDDING_PROVIDER": "Choose from: " + ", ".join(get_embedding_providers()),
                "VECTOR_PROVIDER": "Choose from: " + ", ".join(get_vector_providers()),
                "TEXT_SPLITTER": "Choose from: " + ", ".join(get_text_splitters())
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
