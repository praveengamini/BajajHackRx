import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    BEARER_TOKEN = "f31b509fa84200a558797aa954acd8bc0296cb8c78649676ac6e716c75c15c15"
    
    # Model Configuration
    GEMINI_MODEL_NAME = "gemini-2.0-flash"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    
    # Text Processing Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 5
    
    # API Configuration
    PORT = int(os.environ.get("PORT", 8000))
    HOST = "0.0.0.0"
    
    # Generation Config
    TEMPERATURE = 0.1
    MAX_OUTPUT_TOKENS = 2048
    TOP_P = 0.9
    TOP_K = 40
    
    # ChromaDB Configuration
    CHROMA_DB_PATH = "./chroma_db"