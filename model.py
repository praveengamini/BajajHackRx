import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from typing import Optional, List, Dict, Any
import requests
import chromadb
from chromadb.config import Settings
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import io
from docx import Document
import tempfile


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="LLM-Powered Intelligent Query-Retrieval System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")


class LocalGeminiChatLLM(LLM, BaseModel):
    model_name: str = "gemini-2.0-flash"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY")

    @property
    def _llm_type(self) -> str:
        return "local_gemini_chat_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.gemini_api_key
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,  # Lower temperature for more consistent answers
                "maxOutputTokens": 2048,
                "topP": 0.9,
                "topK": 40
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            if not response.ok:
                print("Gemini API error:", response.text)
                response.raise_for_status()
            
            response_data = response.json()
            
            # Extract text from Gemini API response structure
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"]
            
            # Fallback if structure is different
            return "Sorry, I couldn't generate a proper response."
            
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def download_document(self, url: str) -> bytes:
        """Download document from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract PDF text: {str(e)}")
    
    def extract_text_from_docx(self, docx_content: bytes) -> str:
        """Extract text from DOCX content"""
        try:
            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file.write(docx_content)
                tmp_file.flush()
                doc = Document(tmp_file.name)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract DOCX text: {str(e)}")
    
    def process_document(self, document_url: str) -> FAISS:
        """Process document and create vector store"""
        # Download document
        content = self.download_document(document_url)
        
        # Determine file type and extract text
        if document_url.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(content)
        elif document_url.lower().endswith('.docx'):
            text = self.extract_text_from_docx(content)
        else:
            # Assume it's plain text
            text = content.decode('utf-8', errors='ignore')
        
        # Split text into chunks
        texts = self.text_splitter.create_documents([text])
        
        # Create vector store
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        
        return vectorstore, text


class QueryProcessor:
    def __init__(self, llm: LocalGeminiChatLLM):
        self.llm = llm
    
    def process_query(self, query: str, vectorstore: FAISS, document_text: str) -> str:
        """Process a single query against the document"""
        
        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create enhanced prompt for insurance/legal domain
        prompt = f"""You are an expert in insurance policy analysis and legal document interpretation. 
Your task is to answer questions about policy documents with high accuracy and provide specific details.

DOCUMENT CONTEXT:
{context}

QUERY: {query}

INSTRUCTIONS:
1. Answer the question based ONLY on the provided document context
2. If the answer exists in the document, provide specific details including:
   - Exact conditions, time periods, amounts, percentages
   - Any relevant limitations or exclusions
   - Specific clause references where applicable
3. If the information is not found in the document, clearly state "This information is not available in the provided document"
4. For insurance queries, focus on coverage details, waiting periods, conditions, and exclusions
5. Be precise and factual - avoid speculation or general knowledge
6. Use clear, professional language

ANSWER:"""

        try:
            answer = self.llm._call(prompt)
            return answer.strip()
        except Exception as e:
            return f"Error processing query: {str(e)}"


# Pydantic models for API
class HackRXRunRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]

class HackRXRunResponse(BaseModel):
    answers: List[str]

# Legacy models (keeping for backward compatibility)
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


# Global instances
document_processor = DocumentProcessor()
llm = LocalGeminiChatLLM()
query_processor = QueryProcessor(llm)

# Storage for legacy endpoints
vectorstores = {}
memory_instances = {}


@app.get("/")
def root():
    return {
        "message": "LLM-Powered Intelligent Query-Retrieval System",
        "version": "1.0.0",
        "model": "gemini-2.0-flash",
        "capabilities": ["PDF processing", "DOCX processing", "Semantic search", "Insurance policy analysis"]
    }


@app.post("/hackrx/run", response_model=HackRXRunResponse)
async def hackrx_run(request: HackRXRunRequest):
    """
    Main endpoint for the hackathon submission.
    Processes documents and answers questions with structured JSON responses.
    """
    try:
        # Process the document
        print(f"Processing document: {request.documents}")
        vectorstore, document_text = document_processor.process_document(request.documents)
        
        # Process all questions
        answers = []
        for question in request.questions:
            print(f"Processing question: {question}")
            answer = query_processor.process_query(question, vectorstore, document_text)
            answers.append(answer)
        
        return HackRXRunResponse(answers=answers)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}"
        )


# Legacy endpoints (keeping for backward compatibility)
@app.post("/embed")
async def embed_text(request: EmbedRequest):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.create_documents([request.text])
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstores[request.pdfId] = vectorstore
        return {"message": f"Embedding stored for PDF ID: {request.pdfId}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error embedding text: {str(e)}")


@app.post("/api/generate")
def generate_answer(request: GenerateRequest):
    vectorstore = vectorstores.get(request.pdfId)
    if not vectorstore:
        raise HTTPException(status_code=404, detail="No embeddings found for this PDF ID")
    
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(request.message)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        enhanced_prompt = f"""Context from document:
{context}

Question: {request.message}

Please answer the question based on the provided context. If the answer cannot be found in the context, say so clearly."""
        
        answer = llm._call(enhanced_prompt)
        
        return {
            "answer": answer,
            "source_documents": len(relevant_docs),
            "session_id": request.sessionId
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.get("/health")
def health_check():
    try:
        collections = chroma_client.list_collections()
        collection_count = len(collections)
        
        # Test Gemini API connectivity
        test_llm = LocalGeminiChatLLM()
        try:
            test_response = test_llm._call("Test connection")
            api_status = "connected"
        except:
            api_status = "disconnected"
        
        return {
            "status": "healthy",
            "model": "gemini-2.0-flash",
            "api_status": api_status,
            "loaded_pdf_ids": list(vectorstores.keys()),
            "chromadb_collections": collection_count,
            "capabilities": {
                "document_processing": ["PDF", "DOCX", "TXT"],
                "embedding_model": "all-MiniLM-L6-v2",
                "vector_search": "FAISS",
                "llm_model": "gemini-2.0-flash"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "model": "gemini-2.0-flash"
        }


@app.get("/api/v1/models")
def list_models():
    """API endpoint to list available models"""
    return {
        "models": [
            {
                "id": "gemini-2.0-flash",
                "type": "llm",
                "capabilities": ["text-generation", "document-analysis", "question-answering"]
            },
            {
                "id": "all-MiniLM-L6-v2",
                "type": "embedding",
                "capabilities": ["text-embedding", "semantic-search"]
            }
        ]
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT"))
    uvicorn.run("model:app", host="0.0.0.0", port=port, workers=1)