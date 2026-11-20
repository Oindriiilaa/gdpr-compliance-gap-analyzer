"""
debug_ranking.py
Quick diagnostic: shows the FULL similarity ranking of all policy chunks
for a given question, so we can see if a relevant-but-vague section is
a close miss or genuinely far down the ranking.
"""
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
    "Does our policy explain how customers can request deletion of their data?"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection("compliance_chunks")

q_emb = model.encode([question]).tolist()
res = collection.query(
    query_embeddings=q_emb,
    n_results=9,
    where={"source_type": "company_policy"},
)

print(f"Question: {question}\n")
print("Full ranking of ALL 9 policy sections:\n")
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    sim = 1 - dist
    print(f"{sim:.3f}  {meta['source_label']:20s}  {doc[:70]}")