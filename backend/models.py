# backend/models.py

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# --- 1. Message Model (For history) ---
class Message(BaseModel):
    # 'user' or 'assistant'
    role: str
    content: str

# --- 2. Chat Request Model (Input from Frontend) ---
class ChatRequest(BaseModel):
    # Optional per their contract, but we still accept it for logging
    session_id: str | None = None
    user_id: str | None = None 
    
    # REQUIRED: They need the full history, not just the last message
    messages: list[Message]
    
    # Optional RAG parameters
    top_k: int = 4
    use_reranker: bool = True

# --- 3. Response Models (Output from RAG team) ---

# Structure for the metadata inside each retrieved chunk
class RetrievedMetadata(BaseModel):
    # Assuming these are the key fields for product data
    product_id: str | None = None
    product_name: str | None = None
    price: float | None = None
    # Use Dict[str, Any] if the metadata keys are highly variable
    
class RetrievedChunk(BaseModel):
    metadata: RetrievedMetadata
    snippet: str

# --- 4. Chat Response Model (Output to Frontend) ---
class ChatResponse(BaseModel):
    status: str = "success"
    session_id: str | None = None
    answer: str                         # The final text response
    messages: list[Message]             # The complete, updated chat history
    retrieved: list[RetrievedChunk]     # The source documents (product chunks)

    