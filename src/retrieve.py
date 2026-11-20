"""
retrieve.py
Step 4 of the Compliance RAG pipeline.

Takes a natural-language question, embeds it with the same model used
in embed_and_store.py, and retrieves the most semantically similar
chunks from ChromaDB (a mix of GDPR articles and policy sections).

This script is a standalone TEST of retrieval quality — no LLM involved
yet. Good retrieval is the foundation of a good RAG system, so we
verify this works well before spending API calls on generation.

Usage:
    python src/retrieve.py "Does our policy explain how to request data deletion?"
"""

import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "compliance_chunks"

# How many chunks to retrieve, and from where. We split the request so we
# reliably get BOTH a regulation chunk and a policy chunk, rather than
# risking the top-k being dominated by one source type.
TOP_K_PER_SOURCE = 5


def retrieve(question: str, model, collection):
    query_embedding = model.encode([question]).tolist()

    results = {}
    for source_type in ["regulation", "company_policy"]:
        res = collection.query(
            query_embeddings=query_embedding,
            n_results=TOP_K_PER_SOURCE,
            where={"source_type": source_type},
        )
        results[source_type] = res

    return results


def print_results(results):
    for source_type, res in results.items():
        label = "GDPR (Regulation)" if source_type == "regulation" else "Company Policy"
        print(f"\n--- Top matches from: {label} ---")
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        distances = res["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            similarity = 1 - dist  # cosine distance -> similarity, roughly
            print(f"\n[{meta['source_label']}] (similarity: {similarity:.3f})")
            print(doc[:300])


def main():
    if len(sys.argv) < 2:
        question = "Does our policy explain how customers can request deletion of their data?"
        print(f"No question provided, using default:\n\"{question}\"\n")
    else:
        question = " ".join(sys.argv[1:])
        print(f"Question: \"{question}\"\n")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    results = retrieve(question, model, collection)
    print_results(results)


if __name__ == "__main__":
    main()