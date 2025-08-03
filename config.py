import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    BEARER_TOKEN = os.getenv("BEARER_TOKEN")
    
    GEMINI_MODEL_NAME = "gemini-2.0-flash"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 5
    
    PORT = int(os.environ.get("PORT", 8000))
    HOST = "0.0.0.0"
    
    TEMPERATURE = 0.1
    MAX_OUTPUT_TOKENS = 2048
    TOP_P = 0.9
    TOP_K = 40
    
    # FAISS configuration
    FAISS_INDEX_PATH = "./faiss_indexes"
    FAISS_METADATA_PATH = "./faiss_metadata"