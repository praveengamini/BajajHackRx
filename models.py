from pydantic import BaseModel
from typing import Optional, List

# Main API Models
class HackRXRunRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]

class HackRXRunResponse(BaseModel):
    answers: List[str]

# Legacy API Models (keeping for backward compatibility)
class EmbedRequest(BaseModel):
    text: str
    pdfId: str

class GenerateRequest(BaseModel):
    pdfId: str
    message: str
    sessionId: Optional[str] = "default"

class ClearHistoryRequest(BaseModel):
    pdfId: str
    sessionId: Optional[str] = "default"

# Response Models
class HealthResponse(BaseModel):
    status: str
    model: str
    api_status: Optional[str] = None
    loaded_pdf_ids: List[str] = []
    chromadb_collections: Optional[int] = None
    capabilities: Optional[dict] = None
    error: Optional[str] = None

class GenerateResponse(BaseModel):
    answer: str
    source_documents: int
    session_id: str

class ModelInfo(BaseModel):
    id: str
    type: str
    capabilities: List[str]

class ModelsResponse(BaseModel):
    models: List[ModelInfo]