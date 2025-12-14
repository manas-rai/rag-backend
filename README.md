# RAG Backend with FastAPI and OpenAI

This is a Retrieval-Augmented Generation (RAG) backend implementation using FastAPI and OpenAI. The system allows you to process documents, create embeddings, and query them using natural language.

## Features

- Document processing and chunking
- Vector storage using ChromaDB
- OpenAI integration for embeddings and completions
- FastAPI REST API
- Environment-based configuration
- Dependency injection
- Error handling

## Prerequisites

- Python 3.11 or higher
- `make` (usually pre-installed on macOS/Linux)
- (Optional) `pyenv` for Python version management

## Setup

### Quick Setup (Recommended)

Run the complete setup with a single command:

```bash
make setup
```

This will:
- Create a Python virtual environment (`.venv`)
- Create a `.env` file with default configuration
- Install all dependencies (base + ML + providers)

### Manual Setup Steps

If you prefer to set up manually or need to customize:

1. **Create virtual environment:**
   ```bash
   make venv
   ```

2. **Create `.env` file:**
   ```bash
   make env
   ```

3. **Install dependencies:**
   ```bash
   make install-all    # Install all dependencies
   # or
   make install        # Install only base dependencies
   ```

4. **Edit `.env` file** with your API keys and configuration:
   ```bash
   # Required for Groq (default LLM provider)
   GROQ_API_KEY=your_groq_api_key
   
   # Optional: Other provider API keys
   # OPENAI_API_KEY=your_openai_api_key
   # AZURE_API_KEY=your_azure_api_key
   # PINECONE_API_KEY=your_pinecone_api_key
   ```

### Pyenv Setup (Optional)

If you're using `pyenv` and having issues:

```bash
# Diagnose pyenv setup
make setup-pyenv

# Auto-fix common pyenv issues
make fix-pyenv
```

### Available Makefile Commands

Run `make help` to see all available commands:

- `make setup` - Complete setup (venv + .env + dependencies)
- `make venv` - Create virtual environment
- `make env` - Create .env file with defaults
- `make install` - Install base dependencies only
- `make install-all` - Install all dependencies
- `make run` - Run the FastAPI application
- `make check-python` - Check Python installation
- `make setup-pyenv` - Diagnose pyenv setup
- `make fix-pyenv` - Fix common pyenv issues
- `make clean` - Remove virtual environment and .env file
- `make test` - Run tests (if available)

## Running the Application

### Using Makefile (Recommended)

Start the FastAPI server:
```bash
make run
```

### Manual Run

If you prefer to run manually:

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the server
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Process Documents
- **POST** `/documents`
- **Body**:
```json
{
    "documents": [
        "Your document text here...",
        "Another document text..."
    ]
}
```

### 2. Query
- **POST** `/query`
- **Body**:
```json
{
    "query": "Your question here"
}
```
- **Response**:
```json
{
    "answer": "Generated answer based on context",
    "context": ["Relevant context chunks used for generation"]
}
```

## API Documentation

Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/           # Request/Response models
│   ├── routers/          # API route handlers
│   ├── services/         # Business logic
│   ├── llm/              # LLM provider implementations
│   ├── embedding/        # Embedding provider implementations
│   ├── vector/           # Vector store implementations
│   ├── text/             # Text processing and splitting
│   ├── document/         # Document preprocessing
│   ├── prompts/           # Prompt templates
│   └── utils/            # Utilities and configuration
├── data/
│   └── vector_store/     # Local vector store data
├── requirements-base.txt      # Base dependencies
├── requirements-ml.txt        # ML dependencies
├── requirements-providers.txt # Provider-specific dependencies
├── local-requirements.txt    # Local development dependencies
├── Makefile                  # Setup and build automation
├── Dockerfile                # Docker configuration
├── .env                      # Environment variables (create with `make env`)
└── README.md
```