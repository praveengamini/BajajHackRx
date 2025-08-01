# LLM-Powered Intelligent Query-Retrieval System

A modular FastAPI application for document processing and intelligent question answering using Gemini LLM and vector embeddings.

## 🏗️ Project Structure

```
project/
├── main.py                 # Main FastAPI application
├── config.py              # Configuration settings
├── models.py              # Pydantic models
├── auth.py                # Authentication middleware
├── routes.py              # API routes
├── llm_service.py         # LLM service implementation
├── document_processor.py  # Document processing service
├── query_processor.py     # Query processing service
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-query-retrieval-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your credentials
GEMINI_API_KEY=your_gemini_api_key_here
PORT=8000
```

### 3. Run the Application

```bash
# Development mode
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 API Documentation

### Base URL
- **Local Development**: `http://localhost:8000/api/v1`
- **Authentication**: `Bearer f31b509fa84200a558797aa954acd8bc0296cb8c78649676ac6e716c75c15c15`

### Main Endpoint

#### POST `/api/v1/hackrx/run`
Process documents and answer questions.

**Request:**
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "A grace period of thirty days is provided...",
        "There is a waiting period of thirty-six months..."
    ]
}
```

### Other Endpoints

- `GET /api/v1/health` - Health check
- `GET /api/v1/models` - List available models
- `POST /api/v1/embed` - Embed text (legacy)
- `POST /api/v1/generate` - Generate answer (legacy)

## 🔧 Features

- **Document Processing**: PDF, DOCX, and text file support
- **Vector Search**: FAISS-based semantic search
- **LLM Integration**: Gemini 2.0 Flash model
- **Authentication**: Bearer token authentication
- **Modular Architecture**: Clean separation of concerns
- **Legacy Support**: Backward compatibility with existing endpoints

## 🏛️ Architecture

### Core Components

1. **FastAPI Application** (`main.py`)
   - Main application setup
   - Middleware configuration
   - Route registration

2. **Configuration** (`config.py`)
   - Centralized configuration management
   - Environment variable handling

3. **Authentication** (`auth.py`)
   - Bearer token validation
   - Security middleware

4. **Document Processing** (`document_processor.py`)
   - File download and parsing
   - Text extraction from various formats
   - Vector store creation

5. **Query Processing** (`query_processor.py`)
   - Semantic search
   - Context-aware answer generation

6. **LLM Service** (`llm_service.py`)
   - Gemini API integration
   - Response processing

### Data Flow

1. **Document Upload** → Download → Text Extraction → Chunking → Embedding → Vector Store
2. **Query Processing** → Vector Search → Context Retrieval → LLM Generation → Response

## 🔒 Security

- Bearer token authentication required for all endpoints
- Input validation using Pydantic models
- Error handling and sanitization
- CORS configuration for cross-origin requests

## 🧪 Testing

```bash
# Test the main endpoint
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
     -H "Authorization: Bearer f31b509fa84200a558797aa954acd8bc0296cb8c78649676ac6e716c75c15c15" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://example.com/document.pdf",
       "questions": ["What is covered under this policy?"]
     }'

# Health check
curl -X GET "http://localhost:8000/api/v1/health" \
     -H "Authorization: Bearer f31b509fa84200a558797aa954acd8bc0296cb8c78649676ac6e716c75c15c15"
```

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `PORT` | Server port | 8000 |
| `GEMINI_MODEL_NAME` | Gemini model name | gemini-2.0-flash |
| `EMBEDDING_MODEL_NAME` | Embedding model | all-MiniLM-L6-v2 |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.