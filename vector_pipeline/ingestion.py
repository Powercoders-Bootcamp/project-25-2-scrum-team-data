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
from langchain_community.vectorstores import Chroma
#from langchain_chroma import Chroma


from .settings import (
    DATA_PATH,
    COMBINED_TEXT_COLUMN,
    CHROMA_DIR,
    CHROMA_ARCHIVE,
    COLLECTION_NAME,  # currently "langchain"
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from .config import get_bge_embeddings


# ---------------------------------------------------------------------------
# Data loading + conversion to Documents
# ---------------------------------------------------------------------------


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

        # Optional: prepend product name to page_content for extra weight.
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

    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# Build Chroma vectorstore from scratch (one-time heavy job)
# ---------------------------------------------------------------------------


def build_chroma_vectorstore() -> Chroma:
    """
    End-to-end pipeline to build and persist the Chroma DB from scratch:

      1. Load dataframe.
      2. Convert rows -> Documents.
      3. Chunk Documents.
      4. Embed and build Chroma DB.
      5. Persist to disk (via `persist_directory`).

    This will delete any existing Chroma directory first.

    NOTE: We intentionally do **not** set an explicit `collection_name` here,
    so Chroma uses its default (currently "langchain"), which matches the DB
    built in Muhammet's notebook.
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
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    print("Chroma DB built and stored.")
    print("Document count:", vectorstore._collection.count())
    return vectorstore


# ---------------------------------------------------------------------------
# Load or unzip an existing Chroma vectorstore (runtime path)
# ---------------------------------------------------------------------------


def build_or_load_vectorstore() -> Chroma:
    """
    Load an existing Chroma vectorstore if possible.

    Priority:
      1. If the Chroma directory already exists and is non-empty,
         load it from disk (using the default collection name,
         e.g. 'langchain' as in the shared notebook).
      2. Else, if a zipped archive (chroma_db.zip) exists, unzip it
         into CHROMA_DIR and then load the vectorstore.
      3. If neither directory nor archive exist, build the DB from scratch.

    This avoids re-embedding everything every time someone clones the repo
    and ensures we use the same collection that was created in the
    data teammate's notebook.
    """

    # 1) Existing directory → just load it
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print(f"Loading existing Chroma DB from {CHROMA_DIR} ...")
        embeddings = get_bge_embeddings()
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print("Vectorstore loaded successfully.")
        print("Document count:", vectorstore._collection.count())
        return vectorstore

    # 2) No directory (or empty), but archive exists → unzip & load
    if CHROMA_ARCHIVE.exists():
        print(f"No Chroma directory found, but archive exists at {CHROMA_ARCHIVE}.")
        print("Unpacking Chroma DB archive into CHROMA_DIR ...")

        # Remove any existing directory (empty / wrong)
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(CHROMA_ARCHIVE), extract_dir=str(CHROMA_DIR))

        print("Archive unpacked. Loading vectorstore ...")
        embeddings = get_bge_embeddings()
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print("Vectorstore loaded successfully.")
        print("Document count:", vectorstore._collection.count())
        return vectorstore

    # 3) Nothing exists → build from scratch
    print("No existing Chroma DB directory or archive found. Building Chroma vectorstore ...")
    return build_chroma_vectorstore()


if __name__ == "__main__":
    # Running this file directly will *always* rebuild the DB.
    build_chroma_vectorstore()
