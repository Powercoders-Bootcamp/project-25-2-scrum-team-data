"""
settings.py

Central configuration for paths and chunking parameters used by
the vector pipeline (ingestion + retrieval).
"""
from pathlib import Path

# Directory: .../PROJECT-25-2-SCRUM-TEAM-DATA/vector_pipeline
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT: Path = _THIS_DIR.parent

# Path to the preprocessed dataframe with a `combined_text` column.
DATA_PATH: Path = PROJECT_ROOT / "data" / "processed_product.pkl"

# Where to persist the Chroma DB.
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"

# Optional: zipped Chroma DB archive (for distribution via Git).
CHROMA_ARCHIVE: Path = PROJECT_ROOT / "chroma_db.zip"

# Column that contains the full combined text for each product.
COMBINED_TEXT_COLUMN: str = "combined_text"

# Name for the Chroma collection (arbitrary, but stable).
COLLECTION_NAME: str = "langchain"

# Chunking parameters for RecursiveCharacterTextSplitter.
CHUNK_SIZE: int = 1500
CHUNK_OVERLAP: int = 100

# Embedding model (bi-encoder)
EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"

# Cross-encoder reranker model
RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"

# LLM model for RAG
#LLM_MODEL_NAME: str = "HuggingFaceH4/zephyr-7b-beta"

#LLM_MODEL_NAME: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

LLM_MODEL_NAME = "sshleifer/tiny-gpt2"