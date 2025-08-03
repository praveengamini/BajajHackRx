import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Server Configuration
    HOST = "0.0.0.0"
    PORT = 8000
    
    # Authentication
    BEARER_TOKEN = os.getenv("BEARER_TOKEN", "your-secret-bearer-token")
    
    # Gemini API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key")
    GEMINI_MODEL_NAME = "gemini-1.5-flash"
    
    # Generation Parameters
    TEMPERATURE = 0.1
    MAX_OUTPUT_TOKENS = 2048
    TOP_P = 0.95
    TOP_K = 40
    
    # Text Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TEXT_SEPARATORS = ["\n\n", "\n", " ", ""]
    
    # Retrieval Configuration
    RETRIEVAL_K = 5
    
    # Embedding Model
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # ChromaDB Configuration
    CHROMA_DB_PATH = "./chroma_db"
    
    # Request Configuration
    REQUEST_TIMEOUT = 30