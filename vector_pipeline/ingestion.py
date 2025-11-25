"""
ingestion.py

This module is responsible for:

1. Loading the preprocessed product dataframe from disk.
2. Turning each row into a LangChain `Document` with:
     - `page_content`  = combined text field
     - `metadata`      = all remaining columns (product attrs etc.)
3. Splitting those documents into overlapping text chunks.
4. Building or loading a persisted Chroma vector store.

This file is *offline*:
it only runs when you (re)build the vector database.
"""

import shutil
from typing import List, Dict, Any

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from .settings import (
    DATA_PATH,
    COMBINED_TEXT_COLUMN,
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from .config import get_bge_embeddings


def load_dataframe() -> pd.DataFrame:
    """
    Load the preprocessed product dataframe from DATA_PATH.

    This file should already contain a `combined_text` column
    (or whatever COMBINED_TEXT_COLUMN is set to).
    """
    df = pd.read_pickle(DATA_PATH)

    if COMBINED_TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Expected column '{COMBINED_TEXT_COLUMN}' in {DATA_PATH}, "
            f"but found: {list(df.columns)}"
        )

    # Drop rows with missing combined_text completely.
    df = df.dropna(subset=[COMBINED_TEXT_COLUMN])

    # Replace remaining NaNs with empty strings for safe metadata.
    df = df.fillna("")

    return df


def dataframe_to_documents(df: pd.DataFrame) -> List[Document]:
    """
    Turn each dataframe row into a single LangChain Document.

    - page_content: df[COMBINED_TEXT_COLUMN]
    - metadata    : all other columns in that row + a `row_index` field
                    to identify the original row.

    Having `row_index` is useful later if you want to group chunks back
    into full "products" or original rows.
    """
    docs: List[Document] = []

    for row_index, (_, row) in enumerate(df.iterrows()):
        row_dict: Dict[str, Any] = row.to_dict()
        page_content = str(row_dict.pop(COMBINED_TEXT_COLUMN))
        metadata = row_dict

        # Add stable row identifier to metadata.
        metadata["row_index"] = int(row_index)

        # Optional: you could prepend product name to page_content
        # to give it extra semantic weight.
        # name = metadata.get("product_name") or metadata.get("title")
        # if name:
        #     page_content = f"{name}\n\n{page_content}"

        docs.append(Document(page_content=page_content, metadata=metadata))

    return docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

    - `chunk_size`   controls the max number of characters per chunk.
    - `chunk_overlap` is how many characters are shared between chunks.
    - `add_start_index=True` stores the starting character index of each chunk
      in the metadata under the key "start_index".
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    chunked_docs = splitter.split_documents(documents)
    return chunked_docs


def build_chroma_vectorstore() -> Chroma:
    """
    End-to-end pipeline to build and persist the Chroma DB from scratch:

      1. Load dataframe.
      2. Convert rows -> Documents.
      3. Chunk Documents.
      4. Embed and build Chroma DB.
      5. Persist to disk (via `persist_directory`).

    This will delete any existing Chroma directory first.
    """
    print(f"Loading data from {DATA_PATH} ...")
    df = load_dataframe()
    print(f"Loaded {len(df)} rows.")

    print("Converting rows to Documents ...")
    raw_docs = dataframe_to_documents(df)
    print(f"Created {len(raw_docs)} raw Documents.")

    print(
        f"Chunking documents (chunk_size={CHUNK_SIZE}, "
        f"overlap={CHUNK_OVERLAP}) ..."
    )
    chunked_docs = chunk_documents(raw_docs)
    print(f"After chunking: {len(chunked_docs)} chunks.")

    embeddings = get_bge_embeddings()

    # Remove old DB if it exists
    if CHROMA_DIR.exists():
        print(f"Removing old Chroma directory at {CHROMA_DIR} ...")
        shutil.rmtree(CHROMA_DIR)

    print("Building Chroma DB and persisting to disk ...")
    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print("Chroma DB built and stored.")
    return vectorstore


def build_or_load_vectorstore() -> Chroma:
    """
    If the Chroma directory already exists and is non-empty, load the
    vectorstore from disk. Otherwise, build it from scratch.

    This avoids re-embedding everything every time you run your notebook.
    """
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print(f"Loading existing Chroma DB from {CHROMA_DIR} ...")
        embeddings = get_bge_embeddings()
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
            # You *could* experiment with explicit cosine space:
            # collection_metadata={"hnsw:space": "cosine"},
        )
        print("Vectorstore loaded successfully.")
        return vectorstore

    print("No existing DB found. Building Chroma vectorstore ...")
    return build_chroma_vectorstore()


if __name__ == "__main__":
    # Running this file directly will *always* rebuild the DB.
    build_chroma_vectorstore()
