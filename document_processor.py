import requests
import PyPDF2
import io
import tempfile
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from fastapi import HTTPException
from config import Config
import uuid

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE, 
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
        self.chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
    
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
    
    def process_document(self, document_url: str) -> tuple:
        """Process document and create ChromaDB collection"""
        content = self.download_document(document_url)
        
        if document_url.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(content)
        elif document_url.lower().endswith('.docx'):
            text = self.extract_text_from_docx(content)
        else:
            text = content.decode('utf-8', errors='ignore')
        
        texts = self.text_splitter.create_documents([text])
        
        collection_name = f"doc_{str(uuid.uuid4())[:8]}"
        
        try:
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"source": document_url}
            )
        except Exception:
            collection = self.chroma_client.get_collection(name=collection_name)
        
        documents = [doc.page_content for doc in texts]
        embeddings = self.embeddings.embed_documents(documents)
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids
        )
        
        return collection, text