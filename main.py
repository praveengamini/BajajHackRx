import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from config import Config
from routes import router

# Create FastAPI application
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Advanced document processing and question answering system with authentication",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Legacy route for backward compatibility (without /api/v1 prefix)
@app.post("/hackrx/run")
async def legacy_hackrx_run(request: Request):
    from routes import hackrx_run
    from auth import verify_token
    from fastapi import Security
    from fastapi.security import HTTPBearer
    
    # For legacy support, we'll skip auth validation here
    # In production, you might want to handle this differently
    mock_user = {"token": "legacy", "authenticated": True}
    return await hackrx_run(request, mock_user)

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        workers=1,
        reload=False
    )
