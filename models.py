from pydantic import BaseModel
from typing import List, Optional

class HackRXRunRequest(BaseModel):
    documents: str 
    questions: List[str]

class HackRXRunResponse(BaseModel):
    answers: List[str]

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

class HealthResponse(BaseModel):
    status: str
    model: str
    api_status: str
    loaded_pdf_ids: List[str]
    chromadb_collections: int
    capabilities: dict

class ModelsResponse(BaseModel):
    models: List[dict]