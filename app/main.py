"""FastAPI application entry point for RAG Backend."""

from typing import Dict
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.routers import documents, query

app = FastAPI(
    title="RAG Backend",
    description="Retrieval-Augmented Generation Backend API",
    version="1.0.0"
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

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Welcome to RAG Backend API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
