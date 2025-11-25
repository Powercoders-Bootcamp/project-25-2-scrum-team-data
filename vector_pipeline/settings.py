"""
settings.py

Central configuration for paths and chunking parameters used by
the vector pipeline (ingestion + retrieval).
"""

from pathlib import Path

# Path to the preprocessed dataframe with a `combined_text` column.
DATA_PATH: Path = Path("data/processed_product.pkl")

# Column that contains the full combined text for each product.
COMBINED_TEXT_COLUMN: str = "combined_text"

# Where to persist the Chroma DB.
CHROMA_DIR: Path = Path("chroma_db")

# Name for the Chroma collection (arbitrary, but stable).
COLLECTION_NAME: str = "products"

# Chunking parameters for RecursiveCharacterTextSplitter.
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150
