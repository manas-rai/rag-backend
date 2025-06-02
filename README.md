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

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo  # or your preferred model
VECTOR_STORE_PATH=./data/vector_store
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Running the Application

Start the FastAPI server:
```bash
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
│   ├── config.py
│   └── rag/
│       ├── __init__.py
│       ├── document_processor.py
│       └── openai_client.py
├── data/
│   └── vector_store/
├── requirements.txt
├── .env
└── README.md
```