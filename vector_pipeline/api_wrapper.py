# vector_pipeline/api_wrapper.py

"""
High-level wrapper functions for the product RAG system.

This module is what you give to the *backend team*.

It hides the internal details (Chroma, embeddings, reranker, LLM) and
exposes a small, stable Python API they can call from a web service:

    - run_chat(messages, k=4, use_reranker=True)

Typical usage on backend side:

    from vector_pipeline.api_wrapper import run_chat

    def handle_request(payload):
        messages = payload["messages"]  # list of {"role": ..., "content": ...}
        result = run_chat(messages)

        return {
            "answer": result["answer"],
            "messages": result["messages"],
            "retrieved": result["retrieved"],
        }

The backend is responsible for storing chat history per user (by IP,
session id, database, etc.) and sending the full history in `messages`
when calling run_chat().
"""

from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma

from .ingestion import build_or_load_vectorstore
from .retrieval import retrieve_documents, rag_answer


# ---------------------------------------------------------------------------
# Vectorstore singleton
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
# Public API for backend
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


# ---------------------------------------------------------------------------
# Optional: simple stateless helper for debugging
# ---------------------------------------------------------------------------


def answer_single_turn(
    query: str,
    k: int = 4,
    use_reranker: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function for *one-shot* questions without explicit
    chat history. Can be used in quick scripts or notebooks.

    It internally creates a messages list with one user message and
    delegates to `run_chat`.
    """
    messages = [{"role": "user", "content": query}]
    return run_chat(messages, k=k, use_reranker=use_reranker)