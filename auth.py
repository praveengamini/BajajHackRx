from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config import Config

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify Bearer token for API access"""
    if credentials.credentials != Config.BEARER_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return credentials.credentials

def get_current_user(token: str = Depends(verify_token)):
    """Get current authenticated user"""
    return {"token": token, "authenticated": True}