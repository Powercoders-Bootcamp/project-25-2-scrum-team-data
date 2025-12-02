"""
retrieval.py

Core retrieval + RAG utilities used by notebooks, scripts and chat demos.

Includes:
- load_vectorstore(): load persisted Chroma DB.
- retrieve_documents(): one-shot retrieval with optional cross-encoder reranker.
- retrieve_products(): simple dict-based API for other components.
- rag_answer(): RAG over the product corpus using a Hugging Face LLM.
- Debug helpers to inspect the vector store.
- Optional interactive CLI loops (for terminal / notebook use).

All embeddings / LLMs / reranker are from Hugging Face.
No OpenAI dependency in this module.
"""

from typing import List, Dict, Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .settings import CHROMA_DIR
from .config import get_bge_embeddings, get_bge_reranker, get_hf_llm


# ---------------------------------------------------------------------------
# Base objects (vectorstore + LLM)
# ---------------------------------------------------------------------------


def load_vectorstore() -> Chroma:
    """
    Load the existing Chroma vector store from CHROMA_DIR.

    We deliberately do NOT pass `collection_name` here, so Chroma uses the
    default collection (currently "langchain"), which matches the DB built
    in the data notebook.
    """
    embeddings = get_bge_embeddings()

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        # no collection_name → default "langchain"
    )
    return vectorstore


# Singleton LLM for RAG (Hugging Face).
_llm = get_hf_llm()


def get_llm():
    """Return the shared Hugging Face LLM instance."""
    return _llm


# ---------------------------------------------------------------------------
# Retrieval + reranking
# ---------------------------------------------------------------------------


def retrieve_documents(
    query: str,
    vs: Chroma,
    k: int = 4,
    use_reranker: bool = False,
    initial_k: Optional[int] = None,
) -> List[Document]:
    """
    Retrieve top-k documents for `query` from `vs`.

    If `use_reranker` is False:
        - simple vector similarity search with k nearest neighbors.

    If `use_reranker` is True:
        - first retrieve `initial_k` candidates by vector similarity
          (default: 4 * k).
        - then re-score them with a cross-encoder reranker.
        - finally return top-k reranked results.
    """
    if not use_reranker:
        retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
        return retriever.invoke(query)

    # Two-stage retrieval: vector search, then cross-encoder reranking.
    if initial_k is None:
        initial_k = max(k * 4, k + 8)

    base_retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": initial_k},
    )
    candidate_docs = base_retriever.invoke(query)

    if not candidate_docs:
        return []

    reranker = get_bge_reranker()
    scores = reranker.score(query, [d.page_content for d in candidate_docs])

    # Sort docs by reranker score descending
    pairs = list(zip(candidate_docs, scores))
    pairs.sort(key=lambda t: t[1], reverse=True)

    reranked_docs = [doc for doc, _ in pairs[:k]]
    return reranked_docs


# ---------------------------------------------------------------------------
# Simple API for other parts of the system
# ---------------------------------------------------------------------------


def retrieve_products(
    query: str,
    k: int = 4,
    use_reranker: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convenience function for other parts of the system.
    Returns a list of dicts with metadata + snippet for top-k chunks.
    """
    vs = load_vectorstore()
    docs = retrieve_documents(query, vs, k=k, use_reranker=use_reranker)

    results: List[Dict[str, Any]] = []
    for doc in docs:
        results.append(
            {
                "metadata": doc.metadata,
                "snippet": doc.page_content[:400],
            }
        )
    return results


def summarize_retrieved_products(
    query: str,
    k: int = 3,
    use_reranker: bool = True,
) -> str:
    """
    Offline summary (no LLM): summarize basic info about top-k hits.
    """
    vs = load_vectorstore()
    docs = retrieve_documents(query, vs, k=k, use_reranker=use_reranker)

    if not docs:
        return f"No matching products for query: {query!r}"

    lines: List[str] = []
    lines.append(f"Found {len(docs)} products for: '{query}'\n")

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        title = meta.get("product_name", meta.get("title", "Unknown name"))
        brand = meta.get("brand", "Unknown brand")
        price = meta.get("price", "unknown price")
        lines.append(f"{i}. {title} — {brand}, Price: {price}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RAG over the product corpus
# ---------------------------------------------------------------------------


def rag_answer(
    query: str,
    vs: Optional[Chroma] = None,
    k: int = 4,
    use_reranker: bool = True,
) -> str:
    """
    Retrieve top-k documents and ask an LLM to answer using the context.

    If the answer is not in the context, the LLM is instructed to say it
    doesn't know.
    """
    if vs is None:
        vs = load_vectorstore()

    docs = retrieve_documents(query, vs, k=k, use_reranker=use_reranker)

    if not docs:
        return "I couldn't find anything relevant in the product database."

    context = "\n\n---\n\n".join(doc.page_content for doc in docs[:k])

    prompt = """
Use ONLY the following product information to answer the question.
If the answer is not in the context, say you don't know.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
""".strip().format(context=context, query=query)

    llm = get_llm()
    response = llm.invoke(prompt)
    # HuggingFacePipeline.invoke usually returns a string.
    return response


# ---------------------------------------------------------------------------
# Debug / preview helpers
# ---------------------------------------------------------------------------


def print_product_card(doc: Document) -> None:
    """
    Pretty-print a single product chunk with key metadata fields.
    """
    meta = doc.metadata

    title = meta.get("product_title", "Unknown product")
    product_id = meta.get("product_id", "N/A")
    brand = meta.get("store", "Unknown brand")
    price = meta.get("price", "Unknown price")

    print("──────────────────────────────────────────────")
    print(f"🛍️  {title}")
    print(f"   ID: {product_id}")
    print(f"   Brand: {brand}")
    print(f"   Price: {price}")
    print("   Description snippet:")
    print("   " + doc.page_content[:250].replace("\n", " "))
    print("──────────────────────────────────────────────")


def preview_vectorstore(vs: Chroma, k: int = 5) -> None:
    """
    Retrieve k documents with a dummy query and print their metadata + snippet.

    This is useful just to sanity-check what ended up in the vector store.
    """
    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(" ")

    print(f"Got {len(docs)} documents from vectorstore.\n")
    for i, doc in enumerate(docs, start=1):
        print(f"=== Doc {i} ===")
        print("Metadata:")
        for key, value in doc.metadata.items():
            print(f"  {key}: {value}")
        print("\nText snippet:")
        text = doc.page_content
        print("  " + text[:400] + ("..." if len(text) > 400 else ""))
        print("\n" + "-" * 70 + "\n")


def debug_similarity_search_with_scores(
    vs: Chroma,
    query: str,
    k: int = 10,
) -> None:
    """
    Low-level peek into Chroma: print (doc, score) pairs for a query.
    """
    results = vs.similarity_search_with_score(query=query, k=k)
    print(f"Got {len(results)} results for query: {query!r}\n")

    for i, (doc, score) in enumerate(results, start=1):
        print(f"=== Result {i} ===  (score={score:.4f})")
        print_product_card(doc)


def debug_single_retrieval(
    vs: Chroma,
    query: str,
    k: int = 4,
    use_reranker: bool = True,
) -> None:
    """
    For Jupyter: run a single retrieval for a hardcoded query and print cards.
    """
    docs = retrieve_documents(query, vs, k=k, use_reranker=use_reranker)
    print(f"\nRetrieved {len(docs)} documents for query: {query!r}")
    print(f"Reranker enabled: {use_reranker}\n")

    for doc in docs:
        print_product_card(doc)


def debug_single_rag(
    vs: Chroma,
    query: str,
    k: int = 4,
    use_reranker: bool = True,
) -> None:
    """
    For Jupyter: run a single RAG answer for a hardcoded query.
    """
    print(f"RAG answer for query: {query!r}  (reranker={use_reranker})\n")
    answer = rag_answer(query, vs=vs, k=k, use_reranker=use_reranker)
    print(answer)


# ---------------------------------------------------------------------------
# Interactive CLI loops (optional)
# ---------------------------------------------------------------------------


def interactive_retrieval_chat(vs: Chroma, use_reranker: bool = True) -> None:
    """
    Semantic product search with optional reranking; prints product cards.

    This is *just* for terminal / notebook quick tests.
    """
    print("Semantic Product Search — type 'exit' to stop.\n")
    print(f"Reranker enabled: {use_reranker}\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("\nBot: Goodbye 👋")
            break

        if not query:
            continue

        docs = retrieve_documents(query, vs, k=3, use_reranker=use_reranker)

        if not docs:
            print("Bot: Sorry, I found nothing.\n")
            continue

        print(f"\nBot: I found {len(docs)} related products:\n")
        for doc in docs:
            print_product_card(doc)
        print()


def interactive_rag_chat(vs: Chroma, use_reranker: bool = True) -> None:
    """
    Full RAG chat demo over the product corpus.
    """
    print("RAG Chat — ask about any product! (type 'exit' to stop)\n")
    print(f"Reranker enabled: {use_reranker}\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Bot: Goodbye 👋")
            break

        if not query:
            continue

        answer = rag_answer(query, vs=vs, use_reranker=use_reranker)
        print("\nBot:", answer)
        print("\n" + "-" * 60 + "\n")
