# langgraph_app/config.py

"""
Helpers for the LangGraph layer to get:
- the shared Chroma vectorstore
- the shared Hugging Face LLM
"""

from typing import Optional

from langchain_chroma import Chroma

from vector_pipeline.ingestion import build_or_load_vectorstore
from vector_pipeline.config import get_hf_llm as _get_hf_llm


_vectorstore: Optional[Chroma] = None


def get_vectorstore() -> Chroma:
    """
    Singleton Chroma vectorstore.

    First call: build/load from disk via vector_pipeline.ingestion.
    Later calls: reuse same instance.
    """
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_or_load_vectorstore()
    return _vectorstore


def get_llm():
    """
    Singleton HF LLM (TinyLlama in your current settings.py).
    """
    return _get_hf_llm()
