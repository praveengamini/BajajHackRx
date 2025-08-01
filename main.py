import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import Config
from routes import router

app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Advanced document processing and question answering system with authentication",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.post("/hackrx/run")
async def legacy_hackrx_run(request):
    """Legacy endpoint for backward compatibility"""
    from routes import hackrx_run
    from auth import verify_token
    from fastapi import Security
    from fastapi.security import HTTPBearer
    
    mock_user = {"token": "legacy", "authenticated": True}
    return await hackrx_run(request, mock_user)

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host=Config.HOST, 
        port=Config.PORT or 8000, 
        workers=1,
        reload=False
    )