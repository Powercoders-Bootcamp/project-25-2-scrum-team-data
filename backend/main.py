# backend/main.py

import os
# Disable tokenizers parallelism to avoid fork deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid 
import json
import traceback
#import redis 

# IMPORTANT: Requires the updated Pydantic models from models.py
from .models import ChatRequest, ChatResponse, Message 

# --- RAG Pipeline Integration (V1 Architecture) ---
try:
    # --- CORRECTED IMPORT PATH ---
    # We are importing the function 'ask_question' from the module 'rag_pipeline.rag_pipeline'
    # from rag_pipeline.rag_pipeline import ask_question as rag_ask_question
    from multi_turn_pipeline.rag_pipeline import ask_question as rag_ask_question

    RAG_SERVICE_READY = True
    print("FastAPI server initialized successfully with RAG Service.")
    
except ImportError as e:
    # This error occurs if the path or file name is wrong
    print(f"FATAL ERROR: Could not import 'rag_pipeline.rag_pipeline'. Check file structure and path. Error: {e}")
    RAG_SERVICE_READY = False
except Exception as e:
    print(f"FATAL ERROR during RAG initialization: {e}")
    RAG_SERVICE_READY = False


# Initialize FastAPI App
app = FastAPI(title="Product RAG Chat API")

# Configure CORS (Accept all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# 2. API ENDPOINTS
# ----------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {
        "status": "healthy",
        "rag_service_ready": RAG_SERVICE_READY
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_handler(req: ChatRequest):
    """
    Handles incoming chat requests, routes them to the RAG pipeline,
    and formats the response according to the ChatResponse model.
    """
    if not RAG_SERVICE_READY:
        raise HTTPException(status_code=503, detail="RAG Service is currently unavailable.")
        
    try:
        print(f"\n📥 Incoming request - session_id: {req.session_id}, user_id: {req.user_id}")
        print(f"   Messages count: {len(req.messages)}")
        
        # Validate session_id - it must be provided
        if not req.session_id or not req.session_id.strip():
            raise ValueError("session_id is required and cannot be empty.")
        
        # 1. Extract the user query and RAG parameters
        # The user's query is always the content of the LAST message in the history list.
        if not req.messages:
            raise ValueError("Messages list is empty.")
            
        # Get the content of the last message (the user's current query)
        user_query = req.messages[-1].content 
        print(f"   User query: {user_query[:100]}...")

        # --- CRITICAL FIX: Calling rag_pipeline.ask_question with only the arguments it accepts ---
        # The rag_pipeline.py function accepts (question, k, use_reranker, session_id) and returns a string.
        print(f"\n🔄 Calling ask_question with session_id={req.session_id}")
        html_answer_string = rag_ask_question(
            question=user_query, 
            k=req.top_k, 
            use_reranker=req.use_reranker,
            session_id=req.session_id.strip()  # Ensure no whitespace
        )
        # ---------------------------------------------------------------------------------------
        
        print(f"✅ Got answer from LLM (length: {len(html_answer_string)})")

        # 2. Format the string output into the expected ChatResponse model

        # Create the assistant's message object
        assistant_message = Message(role="assistant", content=html_answer_string)
        
        # Update the message history with the new assistant message
        updated_messages = req.messages + [assistant_message]

        # The V1 architecture does not return retrieved documents, so we pass an empty list
        retrieved_data = []

        response = ChatResponse(
            status="success",
            session_id=req.session_id,
            answer=html_answer_string, # The final answer string
            messages=updated_messages,  # The complete, updated history
            retrieved=retrieved_data    # Empty list for now
        )
        
        print(f"📤 Sending response with {len(updated_messages)} messages\n")
        return response

    except ValueError as e:
        # Log the detailed error on the server side
        print(f"\n❌ Validation Error: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        
        # Return 400 Bad Request for validation errors
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        # Log the detailed error on the server side
        print(f"\n❌ Internal RAG Chat Error: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        
        # Re-raise the exception as a 500 HTTP response with detailed error info
        raise HTTPException(status_code=500, detail=f"Unexpected error while generating response from RAG service: {str(e)}")


# ----------------------------------------------------------------------
# 3. RUNNER (For Local Development)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)