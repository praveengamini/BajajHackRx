from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests
import PyPDF2
import io
from docx import Document
import tempfile
from fastapi import HTTPException
from config import Config

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE, 
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=Config.TEXT_SEPARATORS
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
    
    def download_document(self, url: str) -> bytes:
        """Download document from URL"""
        try:
            response = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
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
    
    def process_document(self, document_url: str) -> tuple[FAISS, str]:
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