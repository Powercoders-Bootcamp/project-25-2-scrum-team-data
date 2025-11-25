"""
query_demo.py

Very small wrapper to run an interactive similarity-search demo
from the command line.

We reuse the shared `interactive_retrieval_chat` from retrieval.py
so logic stays in one place.
"""

from .ingestion import build_or_load_vectorstore
from .retrieval import interactive_retrieval_chat


def interactive_query_loop() -> None:
    """
    Legacy-compatible entrypoint name for an interactive retrieval demo.
    """
    vs = build_or_load_vectorstore()
    interactive_retrieval_chat(vs, use_reranker=True)


if __name__ == "__main__":
    interactive_query_loop()
