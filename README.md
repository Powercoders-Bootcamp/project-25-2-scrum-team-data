# Vector Pipeline & Retrieval System

A modular embedding–retrieval pipeline for processing product and order data, generating vector representations, storing them in a Chroma database, and running semantic search with optional reranking.

## Overview

The system consists of:
- **Data processing** (cleaning, merging, normalization)
- **Embedding generation**
- **Vector storage in ChromaDB**
- **Similarity search**
- **Optional reranking with cross-encoder**
- **Notebook demos for exploration**

## Models

- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`  
- **Reranker Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

Both models are lightweight and optimized for fast inference while maintaining good semantic performance.

## Repository Structure

project-25-2-scrum-team-data/  
│  
├── **data/**  
│   ├── Product_Information_Dataset.csv  
│   ├── Order_Data_Dataset.csv  
│   └── processed_product.pkl – processed/cleaned product data  
│  
├── **vector_pipeline/**  
│   ├── `config.py` – central configuration (models, embedding dims, DB paths)  
│   ├── `settings.py` – environment + constants loaded by all modules  
│   ├── `ingestion.py` – loads raw data, preprocesses it, generates embeddings, populates Chroma  
│   ├── `retrieval.py` – vector search, optional cross-encoder reranking, result formatting  
│   ├── `query_demo.py` – command-line demo for testing queries  
│   └── `__init__.py`  
│  
├── **chroma_db/** *(ignored in Git)*  
│   Contains the local Chroma SQLite database with all stored embeddings.  
│  
├── `main.ipynb` – notebook demonstrating the full pipeline  
└── `README.md`

## Installation

```bash
pip install -r requirements.txt
```

## How the Pipeline Works

1. **Ingestion**
   - Reads product/order datasets  
   - Cleans and normalizes fields  
   - Generates embeddings using MiniLM  
   - Writes vectors and metadata into Chroma  

2. **Retrieval**
   - Performs approximate nearest-neighbor search in Chroma  
   - Optionally applies reranking using the cross-encoder  
   - Returns matched items with similarity + rerank scores  

3. **Demo Notebook**
   - Shows ingestion, querying, inspection of results, and pipeline behavior  

## Usage

Run data ingestion and embedding creation:

```bash
python vector_pipeline/ingestion.py
```

Run a sample query:

```bash
python vector_pipeline/query_demo.py "your query text"
```

Or explore interactively:

```bash
jupyter notebook main.ipynb
```

## Notes

- `chroma_db/` is excluded from the repository due to size constraints.  
- Regenerate the vector store by re-running the ingestion script.  
- Configuration (model names, paths, thresholds) is controlled centrally in `config.py`.

## License

MIT License.
