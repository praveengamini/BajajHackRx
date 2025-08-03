from langchain.llms.base import LLM

from typing import Optional, List
import requests
from fastapi import HTTPException
from config import Config

class LocalGeminiChatLLM(LLM):
    model_name: str = Config.GEMINI_MODEL_NAME
    gemini_api_key: str = Config.GEMINI_API_KEY

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs.get('model_name', Config.GEMINI_MODEL_NAME)
        self.gemini_api_key = kwargs.get('gemini_api_key', Config.GEMINI_API_KEY)

    @property
    def _llm_type(self) -> str:
        return "local_gemini_chat_llm"

    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> str:
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
                "temperature": Config.TEMPERATURE,
                "maxOutputTokens": Config.MAX_OUTPUT_TOKENS,
                "topP": Config.TOP_P,
                "topK": Config.TOP_K
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

def test_llm_connectivity():
    """Test connectivity to Gemini API"""
    try:
        llm = LocalGeminiChatLLM()
        test_response = llm._call("Hello")
        return "connected"
    except Exception as e:
        return f"error: {str(e)}"