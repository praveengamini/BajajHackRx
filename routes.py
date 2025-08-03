from fastapi import APIRouter, HTTPException, Depends
from typing import List
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from config import Config
from models import (
    HackRXRunRequest, HackRXRunResponse, EmbedRequest, GenerateRequest,
    ClearHistoryRequest, HealthResponse, ModelsResponse
)
from auth import get_current_user
from document_processor import DocumentProcessor
from query_processor import QueryProcessor
from llm_service import LocalGeminiChatLLM, test_llm_connectivity
import uuid
import pickle

router = APIRouter(prefix="/api/v1")

document_processor = DocumentProcessor()
llm = LocalGeminiChatLLM()
query_processor = QueryProcessor(llm)

# In-memory storage for vector stores and document metadata
faiss_vector_stores = {}
document_metadata = {}

@router.get("/")
def root():
    return {
        "message": "LLM-Powered Intelligent Query-Retrieval System with FAISS",
        "version": "1.0.0",
        "model": Config.GEMINI_MODEL_NAME,
        "vector_store": "FAISS",
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
        vector_store, document_text, doc_id = document_processor.process_document(request.documents)
        
        # Store the vector store temporarily for this session
        temp_store_key = f"temp_{doc_id}"
        faiss_vector_stores[temp_store_key] = vector_store
        
        answers = []
        for question in request.questions:
            print(f"Processing question: {question}")
            answer = query_processor.process_query(question, vector_store, document_text)
            answers.append(answer)
        
        # Clean up temporary storage
        if temp_store_key in faiss_vector_stores:
            del faiss_vector_stores[temp_store_key]
        
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
        
        # Split text into chunks
        text_chunks = text_splitter.split_text(request.text)
        
        # Create Langchain documents
        documents = [LangchainDocument(page_content=chunk) for chunk in text_chunks]
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
        
        # Create FAISS vector store
        vector_store = FAISS.from_documents(documents, embeddings)
        
        # Store in memory
        faiss_vector_stores[request.pdfId] = vector_store
        document_metadata[request.pdfId] = {
            "pdf_id": request.pdfId,
            "num_chunks": len(text_chunks),
            "text_length": len(request.text)
        }
        
        # Save to disk
        index_path = os.path.join(Config.FAISS_INDEX_PATH, f"legacy_{request.pdfId}")
        vector_store.save_local(index_path)
        
        # Save metadata
        metadata_path = os.path.join(Config.FAISS_METADATA_PATH, f"legacy_{request.pdfId}.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(document_metadata[request.pdfId], f)
        
        return {"message": f"Embedding stored for PDF ID: {request.pdfId}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error embedding text: {str(e)}")

@router.post("/generate")
async def generate_answer(
    request: GenerateRequest,
    current_user: dict = Depends(get_current_user)
):
    # Try to get vector store from memory first
    vector_store = faiss_vector_stores.get(request.pdfId)
    
    # If not in memory, try to load from disk
    if not vector_store:
        try:
            index_path = os.path.join(Config.FAISS_INDEX_PATH, f"legacy_{request.pdfId}")
            if os.path.exists(index_path):
                embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
                vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
                faiss_vector_stores[request.pdfId] = vector_store
            else:
                raise HTTPException(status_code=404, detail="No embeddings found for this PDF ID")
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"No embeddings found for this PDF ID: {str(e)}")
    
    try:
        # Perform similarity search
        similar_docs = vector_store.similarity_search(
            request.message,
            k=3
        )
        
        if similar_docs:
            context = "\n\n".join([doc.page_content for doc in similar_docs])
        else:
            context = "No relevant context found."
        
        enhanced_prompt = f"""Context from document:
{context}

Question: {request.message}

Please answer the question based on the provided context. If the answer cannot be found in the context, say so clearly."""
        
        answer = llm._call(enhanced_prompt)
        
        return {
            "answer": answer,
            "source_documents": len(similar_docs),
            "session_id": request.sessionId
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@router.get("/health", response_model=HealthResponse)
def health_check():
    try:
        # Count FAISS indexes
        faiss_index_count = 0
        if os.path.exists(Config.FAISS_INDEX_PATH):
            faiss_index_count = len([d for d in os.listdir(Config.FAISS_INDEX_PATH) 
                                   if os.path.isdir(os.path.join(Config.FAISS_INDEX_PATH, d))])
        
        api_status = test_llm_connectivity()
        
        # Get list of loaded PDF IDs
        loaded_pdf_ids = list(faiss_vector_stores.keys())
        
        return HealthResponse(
            status="healthy",
            model=Config.GEMINI_MODEL_NAME,
            api_status=api_status,
            loaded_pdf_ids=loaded_pdf_ids,
            faiss_indexes=faiss_index_count,
            capabilities={
                "document_processing": ["PDF", "DOCX", "TXT"],
                "embedding_model": Config.EMBEDDING_MODEL_NAME,
                "vector_search": "FAISS",
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

@router.delete("/clear/{pdf_id}")
def clear_document(
    pdf_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Clear a specific document from memory and disk"""
    try:
        # Remove from memory
        if pdf_id in faiss_vector_stores:
            del faiss_vector_stores[pdf_id]
        if pdf_id in document_metadata:
            del document_metadata[pdf_id]
        
        # Remove from disk
        index_path = os.path.join(Config.FAISS_INDEX_PATH, f"legacy_{pdf_id}")
        if os.path.exists(index_path):
            import shutil
            shutil.rmtree(index_path)
        
        metadata_path = os.path.join(Config.FAISS_METADATA_PATH, f"legacy_{pdf_id}.pkl")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        return {"message": f"Document {pdf_id} cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing document: {str(e)}")

@router.get("/documents")
async def list_documents(current_user: dict = Depends(get_current_user)):
    """List all processed documents"""
    try:
        documents = document_processor.list_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")