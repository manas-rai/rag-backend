"""FastAPI application entry point for RAG Backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import documents, query
from app.config import get_settings

app = FastAPI(
    title="RAG Backend",
    description="Retrieval-Augmented Generation Backend API",
    version="1.0.0"
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Include routers
app.include_router(documents.router)
app.include_router(query.router)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to RAG Backend API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
