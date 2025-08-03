import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from config import Config
from routes import router

app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Advanced document processing and question answering system with authentication",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include main router
app.include_router(router)

# Legacy endpoint
@app.post("/hackrx/run")
async def legacy_hackrx_run(request: Request):
    from routes import hackrx_run
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
