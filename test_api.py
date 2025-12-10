import httpx
import sys
from typing import Dict, Any, List

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/api/chat"
TIMEOUT_SECONDS = 30.0 # RAG calls can take a while

# --- Teammate's Test Payload (Must be encapsulated in the full ChatRequest structure) ---
USER_QUERY = "I want to have a YAMAHA electric guitar which is the most rated product in the site."

TEST_PAYLOAD = {
    "session_id": "S12345",
    "user_id": "U9001",
    # The ChatRequest model requires the user message to be inside the 'messages' array.
    "messages": [
        {"role": "user", "content": USER_QUERY}
    ],
    "top_k": 10,
    "use_reranker": False
}


def test_chat_endpoint_with_notebook_payload():
    """
    Sends the specific payload from the rag_pipeline_test.ipynb notebook 
    to validate the end-to-end API routing for the V1 architecture.
    """
    # FINAL ATTEMPT to eliminate invisible character issue by using older string formatting
    print("Query: \"%s\"" % USER_QUERY)
    
    try:
        # Use httpx to make the POST request to the running server
        response = httpx.post(API_URL, json=TEST_PAYLOAD, timeout=TIMEOUT_SECONDS)
        
        # 1. Check HTTP Status (Raises exception for 4xx/5xx errors)
        response.raise_for_status()
        
        # 2. Parse JSON response
        data = response.json() # REMOVING TYPE HINT TO BYPASS PERSISTENT SYNTAX ERROR
        
        # 3. Critical Data Assertions
        
        # A. Status Check
        assert data.get("status") == "success", f"API status was not 'success': {data.get('status')}"
        
        # B. Answer Check
        answer = data.get("answer", "")
        print(f"\n✅ Answer received (first 100 chars): {answer[:100]}...")
        # Since this query targets specific product data, we expect a robust answer.
        assert len(answer) > 50, "Answer is suspiciously short (expected detailed product info)."
        
        # C. History Check
        messages: List[Dict[str, str]] = data.get("messages", [])
        assert len(messages) == 2, f"History length mismatch. Expected 2 messages (user + assistant), got {len(messages)}"
        assert messages[-1]['role'] == 'assistant', "The last message is not from the assistant"
        
        # D. Retrieved Documents Check 
        retrieved: List[Any] = data.get("retrieved", [])
        # Since the 'rag_pipeline.ask_question()' V1 signature doesn't return retrieved documents, 
        # main.py correctly sets this to an empty list.
        assert retrieved == [], f"Retrieved documents should be empty for V1, got {retrieved}"

        print("\n✅ *** INTEGRATION TEST SUCCESSFUL ***")
        print("FastAPI successfully processed the notebook payload and returned a valid ChatResponse.")
        
    except httpx.HTTPStatusError as e:
        print(f"\n❌ *** TEST FAILED (HTTP Error) ***")
        print(f"Server responded with status {e.response.status_code}.")
        if e.response.content:
            try:
                print(f"Server Detail: {e.response.json()}")
            except:
                print(f"Server Detail (Raw): {e.response.text}")
        sys.exit(1)
        
    except httpx.ConnectError:
        print(f"\n❌ *** TEST FAILED (Connection Error) ***")
        print(f"Could not connect to the server at {API_URL}. Is 'uvicorn backend.main:app' running in another terminal?")
        sys.exit(1)
        
    except AssertionError as e:
        print(f"\n❌ *** TEST FAILED (Assertion Error) ***")
        print(f"Validation failed: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ *** TEST FAILED (Unexpected Error) ***")
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


# Run the test
if __name__ == "__main__":
    print(f"--- Running Integration Test with Specific Notebook Payload ---")
    test_chat_endpoint_with_notebook_payload()