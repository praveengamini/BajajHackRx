from fastapi import APIRouter, HTTPException, Depends
from typing import List
import chromadb
from config import Config
from models import (
    HackRXRunRequest, HackRXRunResponse, EmbedRequest, GenerateRequest,
    ClearHistoryRequest, HealthResponse, ModelsResponse
)
from auth import get_current_user
from document_processor import DocumentProcessor
from query_processor import QueryProcessor
from llm_service import LocalGeminiChatLLM, test_llm_connectivity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import uuid

router = APIRouter(prefix="/api/v1")

document_processor = DocumentProcessor()
llm = LocalGeminiChatLLM()
query_processor = QueryProcessor(llm)

chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)

chroma_collections = {}
memory_instances = {}

@router.get("/")
def root():
    return {
        "message": "LLM-Powered Intelligent Query-Retrieval System",
        "version": "1.0.0",
        "model": Config.GEMINI_MODEL_NAME,
        "capabilities": ["PDF processing", "DOCX processing", "Semantic search", "Insurance policy analysis"]
    }

@router.post("/hackrx/run", response_model=HackRXRunResponse)
async def hackrx_run(
    request: HackRXRunRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Main endpoint for the hackathon submission.
    Processes documents and answers questions with structured JSON responses.
    """
    try:
        print(f"Processing document: {request.documents}")
        collection, document_text = document_processor.process_document(request.documents)
        
        answers = []
        for question in request.questions:
            print(f"Processing question: {question}")
            answer = query_processor.process_query(question, collection, document_text)
            answers.append(answer)
        
        return HackRXRunResponse(answers=answers)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}"
        )

@router.post("/embed")
async def embed_text(
    request: EmbedRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE, 
            chunk_overlap=100
        )
        texts = text_splitter.create_documents([request.text])
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
        
        collection_name = f"legacy_{request.pdfId}"
        try:
            collection = chroma_client.create_collection(name=collection_name)
        except Exception:
            collection = chroma_client.get_collection(name=collection_name)
        
        documents = [doc.page_content for doc in texts]
        embeddings_list = embeddings.embed_documents(documents)
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        collection.add(
            documents=documents,
            embeddings=embeddings_list,
            ids=ids
        )
        
        chroma_collections[request.pdfId] = collection
        return {"message": f"Embedding stored for PDF ID: {request.pdfId}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error embedding text: {str(e)}")

@router.post("/generate")
def generate_answer(
    request: GenerateRequest,
    current_user: dict = Depends(get_current_user)
):
    collection = chroma_collections.get(request.pdfId)
    if not collection:
        raise HTTPException(status_code=404, detail="No embeddings found for this PDF ID")
    
    try:
        results = collection.query(
            query_texts=[request.message],
            n_results=3
        )
        
        if results['documents'] and len(results['documents']) > 0:
            context = "\n\n".join(results['documents'][0])
        else:
            context = "No relevant context found."
        
        enhanced_prompt = f"""Context from document:
{context}

Question: {request.message}

Please answer the question based on the provided context. If the answer cannot be found in the context, say so clearly."""
        
        answer = llm._call(enhanced_prompt)
        
        return {
            "answer": answer,
            "source_documents": len(results['documents'][0]) if results['documents'] else 0,
            "session_id": request.sessionId
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@router.get("/health", response_model=HealthResponse)
def health_check():
    try:
        collections = chroma_client.list_collections()
        collection_count = len(collections)
        
        api_status = test_llm_connectivity()
        
        return HealthResponse(
            status="healthy",
            model=Config.GEMINI_MODEL_NAME,
            api_status=api_status,
            loaded_pdf_ids=list(chroma_collections.keys()),
            chromadb_collections=collection_count,
            capabilities={
                "document_processing": ["PDF", "DOCX", "TXT"],
                "embedding_model": Config.EMBEDDING_MODEL_NAME,
                "vector_search": "ChromaDB",
                "llm_model": Config.GEMINI_MODEL_NAME
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/models", response_model=ModelsResponse)
def list_models():
    """API endpoint to list available models"""
    return ModelsResponse(
        models=[
            {
                "id": Config.GEMINI_MODEL_NAME,
                "type": "llm",
                "capabilities": ["text-generation", "document-analysis", "question-answering"]
            },
            {
                "id": Config.EMBEDDING_MODEL_NAME,
                "type": "embedding",
                "capabilities": ["text-embedding", "semantic-search"]
            }
        ]
    )