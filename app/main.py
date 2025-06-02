from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import documents, query
from .config import get_settings

app = FastAPI(
    title="RAG Backend",
    description="Retrieval-Augmented Generation Backend API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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