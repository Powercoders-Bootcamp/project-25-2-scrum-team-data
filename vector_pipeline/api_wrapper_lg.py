# vector_pipeline/api_wrapper.py

"""
High-level wrapper functions for the product RAG system.

This module is what you give to the *backend team*.

It hides the internal details (Chroma, embeddings, reranker, LLM) and
exposes two families of APIs:

1) Classic, WITHOUT LangGraph  (backend manages history)
   ------------------------------------------------------
   - run_chat(messages, k=4, use_reranker=True)
   - answer_single_turn(query, k=4, use_reranker=True)

   Typical usage:

       from vector_pipeline.api_wrapper import run_chat

       def handle_request(payload):
           messages = payload["messages"]  # full history
           result = run_chat(messages)

           return {
               "answer": result["answer"],
               "messages": result["messages"],
               "retrieved": result["retrieved"],
           }

2) With LangGraph (internal graph + memory)
   ----------------------------------------
   - run_chat_langgraph_session(user_message, session_id)
       -> LangGraph keeps history per session_id (thread_id).
   - run_chat_langgraph_stateless(messages)
       -> Backend sends full history each time; graph is purely stateless.

   Typical usage (stateful):

       from vector_pipeline.api_wrapper import run_chat_langgraph_session

       def handle_request(payload, ip):
           result = run_chat_langgraph_session(
               user_message=payload["message"],
               session_id=ip,   # or some UUID / cookie
           )
           return result

All variants return a dict with keys:
    - "answer":   assistant message text (string)
    - "messages": full history as list[{"role","content"}]
    - "retrieved": list of context chunks:
          {
              "metadata": {...},   # original product metadata
              "snippet": "..."     # first ~400 characters of text
          }
"""

from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma

from .ingestion import build_or_load_vectorstore
from .retrieval import retrieve_documents, rag_answer

# LangGraph-based helpers live in the separate `langgraph_app` package.
# They are optional: if you don't create langgraph_app, only the classic
# functions will work.
try:
    from langgraph_app import (
        run_chat_session as _lg_run_chat_session,
        run_chat_stateless as _lg_run_chat_stateless,
    )

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Vectorstore singleton (used by classic, non-LangGraph flow)
# ---------------------------------------------------------------------------

_vectorstore: Optional[Chroma] = None


def get_vectorstore() -> Chroma:
    """
    Return a singleton Chroma vectorstore instance.

    - First call: build or load the persisted DB from disk using
      `build_or_load_vectorstore()`.
    - Later calls: reuse the already loaded in-memory instance.
    """
    global _vectorstore

    if _vectorstore is None:
        _vectorstore = build_or_load_vectorstore()
    return _vectorstore


# ---------------------------------------------------------------------------
# Helper: extract the latest user query
# ---------------------------------------------------------------------------


def _get_last_user_message(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Return the last message from the list and assume it is the new user
    message that should be answered.

    The backend should ensure that `messages` is not empty and that the
    last element has `"role": "user"`.
    """
    if not messages:
        raise ValueError("messages list is empty – cannot answer without input.")

    return messages[-1]


# ---------------------------------------------------------------------------
# Classic API for backend (WITHOUT LangGraph)
# ---------------------------------------------------------------------------


def run_chat(
    messages: List[Dict[str, Any]],
    k: int = 4,
    use_reranker: bool = True,
) -> Dict[str, Any]:
    """
    Main entrypoint for the chatbot (WITHOUT LangGraph).

    Parameters
    ----------
    messages:
        Full chat history as a list of dicts. Each dict should have at least:
            - "role": "user" | "assistant" | "system"
            - "content": str

        The last element must be the *new* user message you want to answer.

    k:
        Number of top chunks to use for RAG (default: 4).

    use_reranker:
        Whether to use the BGE cross-encoder reranker in addition to
        vector similarity search. Default: True.

    Returns
    -------
    result: dict with keys
        - "answer": assistant answer text (string)
        - "messages": updated message history (with assistant appended)
        - "retrieved": list of retrieved chunks used as context, each:
              {
                  "metadata": {...},   # original product metadata
                  "snippet": "..."     # first ~400 characters of text
              }
    """
    vs = get_vectorstore()

    last_msg = _get_last_user_message(messages)
    query = str(last_msg.get("content", ""))

    # 1) Retrieve docs (with optional reranker)
    docs = retrieve_documents(
        query=query,
        vs=vs,
        k=k,
        use_reranker=use_reranker,
    )

    retrieved: List[Dict[str, Any]] = []
    for doc in docs:
        retrieved.append(
            {
                "metadata": doc.metadata,
                "snippet": doc.page_content[:400],
            }
        )

    # 2) Produce RAG answer using the same vectorstore and settings
    answer_text = rag_answer(
        query=query,
        vs=vs,
        k=k,
        use_reranker=use_reranker,
    )

    # 3) Append assistant message to history
    assistant_message: Dict[str, Any] = {
        "role": "assistant",
        "content": answer_text,
    }
    updated_messages = [*messages, assistant_message]

    return {
        "answer": answer_text,
        "messages": updated_messages,
        "retrieved": retrieved,
    }


def answer_single_turn(
    query: str,
    k: int = 4,
    use_reranker: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function for *one-shot* questions without explicit
    chat history. Uses the classic, non-LangGraph flow.

    It internally creates a messages list with one user message and
    delegates to `run_chat`.
    """
    messages = [{"role": "user", "content": query}]
    return run_chat(messages, k=k, use_reranker=use_reranker)


# ---------------------------------------------------------------------------
# LangGraph-based APIs for backend
# ---------------------------------------------------------------------------


def _ensure_langgraph():
    if not _LANGGRAPH_AVAILABLE:
        raise RuntimeError(
            "LangGraph-based functions are not available because "
            "`langgraph_app` could not be imported. "
            "Make sure you created the `langgraph_app` package and "
            "that it is on PYTHONPATH."
        )


def run_chat_langgraph_session(
    user_message: str,
    session_id: str,
) -> Dict[str, Any]:
    """
    LangGraph-based, **stateful** chat.

    - Backend sends only the *new* user message + a session_id.
    - LangGraph (via InMemorySaver) keeps the full history internally
      keyed by that session_id (thread_id).

    Parameters
    ----------
    user_message:
        Latest user text input.

    session_id:
        Identifier for this conversation (IP, cookie, UUID, etc.).
        All calls with the same session_id share the same chat history.

    Returns
    -------
    dict with keys:
        - "answer":   assistant answer text (string)
        - "messages": full history as list[{"role","content"}]
        - "retrieved": last retrieved chunks for this turn
    """
    _ensure_langgraph()
    return _lg_run_chat_session(user_message=user_message, session_id=session_id)


def run_chat_langgraph_stateless(
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    LangGraph-based, **stateless** chat.

    - Backend sends full chat history each time (list of dicts with
      "role" and "content").
    - LangGraph runs the RAG agent over that history without using
      internal memory.

    This is useful if:
        - You want to inspect/experiment with the graph behaviour,
        - Or you prefer to store history in your own DB.

    Returns
    -------
    dict with the same structure as other wrappers:
        - "answer":   assistant answer text (string)
        - "messages": full history including the new assistant reply
        - "retrieved": last retrieved chunks for this turn
    """
    _ensure_langgraph()
    return _lg_run_stateless(messages)
