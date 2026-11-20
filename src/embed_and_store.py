"""
embed_and_store.py
Step 3 of the Compliance RAG pipeline.

Loads chunks.json (from chunking.py), generates embeddings using a free
local model (sentence-transformers), and stores them in a local ChromaDB
vector database with metadata for later retrieval + citation.

Run this AFTER chunking.py, and only once (or whenever chunks.json changes) —
it builds a persistent local DB at ./chroma_db so you don't re-embed every run.

Requirements (run locally, needs internet the first time to download the model):
    pip install sentence-transformers chromadb
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

# all-MiniLM-L6-v2: small, fast, free, good enough for a project of this scale
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "compliance_chunks"


def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks. Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Fresh start each time this script runs, to avoid duplicate entries
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # Explicitly use cosine distance so similarity scores are interpretable
    # (ChromaDB defaults to L2/Euclidean, which isn't 1-minus-distance friendly)
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "source_type": c["source_type"],
                "source_label": c["source_label"],
                "title": c["title"],
            }
            for c in chunks
        ],
    )

    print(f"Stored {collection.count()} chunks in ChromaDB at {CHROMA_DIR}")


if __name__ == "__main__":
    main()