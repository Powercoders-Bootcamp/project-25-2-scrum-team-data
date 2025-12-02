"""
config.py

Model configuration utilities for the vector pipeline:

- get_bge_embeddings():
    Sentence encoder for building & querying Chroma.

- CrossEncoderReranker + get_bge_reranker():
    Optional second-stage reranker using a cross-encoder model.

- get_hf_llm():
    Hugging Face based LLM (via transformers + LangChain HuggingFacePipeline)
    for RAG answers. No OpenAI dependency.
"""

from typing import List, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
)
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

from .settings import EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, LLM_MODEL_NAME


# ---------------------------------------------------------------------------
# Embeddings (bi-encoder)
# ---------------------------------------------------------------------------


def get_bge_embeddings() -> HuggingFaceEmbeddings:
    """
    Create a HuggingFace embeddings object for the configured embedding model.

    We match Muhammet's configuration:
    - model_name: EMBEDDING_MODEL_NAME (e.g. "BAAI/bge-base-en-v1.5")
    - device: "cuda" if available, else "cpu"
    - normalize_embeddings=True (recommended for cosine similarity)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """
    Simple cross-encoder reranker wrapper using HF Transformers.

    Usage:
        reranker = CrossEncoderReranker()
        scores = reranker.score("query text", ["doc1", "doc2", ...])

    Higher score = more relevant.
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, docs: List[str]) -> List[float]:
        """
        Return a list of relevance scores (one per doc) for the given query.
        """
        if not docs:
            return []

        queries = [query] * len(docs)

        inputs = self.tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits  # [batch, 1] or [batch]
        scores = logits.squeeze(-1).tolist()

        if isinstance(scores, float):
            scores = [scores]

        return scores


_reranker_instance: Optional[CrossEncoderReranker] = None


def get_bge_reranker() -> CrossEncoderReranker:
    """
    Return a singleton instance of the BGE cross-encoder reranker.

    Avoids re-loading model weights for each request.
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker(RERANKER_MODEL_NAME)
    return _reranker_instance


# ---------------------------------------------------------------------------
# Hugging Face LLM for RAG
# ---------------------------------------------------------------------------


_llm_instance: Optional[HuggingFacePipeline] = None


def get_hf_llm() -> HuggingFacePipeline:
    """
    Return a singleton LangChain LLM built on a Hugging Face
    text-generation pipeline.

    This replaces any OpenAI Chat model usage for the vector_pipeline part.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.0,
        device=device,
    )

    _llm_instance = HuggingFacePipeline(pipeline=gen_pipe)
    return _llm_instance
