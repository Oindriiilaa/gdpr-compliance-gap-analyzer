"""
generate.py
Step 5 of the Compliance RAG pipeline.

Takes a question, retrieves relevant GDPR + policy chunks (reusing
retrieve.py's logic), then sends them to Gemini with a carefully
constrained prompt: answer ONLY using the provided context, cite
the exact article/section, and explicitly flag if the policy is
silent on the topic.

Requirements:
    pip install google-genai
    Get a FREE API key from Google AI Studio: https://aistudio.google.com/apikey
    Set it as an environment variable:
        Windows (PowerShell):  $env:GOOGLE_API_KEY="your-key-here"
        Mac/Linux:              export GOOGLE_API_KEY="your-key-here"

Usage:
    python src/generate.py "Does our policy explain how customers can request deletion of their data?"
"""

import os
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "compliance_chunks"
TOP_K_PER_SOURCE = 6

# Free tier on Google AI Studio as of this writing. If you hit rate limits,
# try "gemini-2.5-flash-lite" which has an even more generous free quota.
GEMINI_MODEL = "gemini-3.5-flash-lite"


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


def build_context_block(results):
    """Turn retrieval results into a clearly labeled text block for the prompt."""
    lines = []

    lines.append("=== RELEVANT GDPR ARTICLES ===")
    for doc, meta in zip(results["regulation"]["documents"][0], results["regulation"]["metadatas"][0]):
        lines.append(f"[{meta['source_label']}] {doc}")

    lines.append("\n=== RELEVANT COMPANY POLICY SECTIONS ===")
    for doc, meta in zip(results["company_policy"]["documents"][0], results["company_policy"]["metadatas"][0]):
        lines.append(f"[{meta['source_label']}] {doc}")

    return "\n\n".join(lines)


SYSTEM_PROMPT = """You are a compliance analysis assistant. You will be given
excerpts from GDPR articles and excerpts from a company's privacy policy.

Your job:
1. Answer the user's question about whether the policy addresses a specific
   GDPR requirement.
2. Base your answer ONLY on the provided excerpts — do not use outside
   knowledge of GDPR or assume anything not stated in the text.
3. Always cite the specific article/section you're referencing, e.g.
   "(GDPR Article 17)" or "(Policy Section 6)".
4. If the policy excerpts do NOT clearly address the question, say so
   explicitly and label it as a COMPLIANCE GAP. Do not be lenient — vague
   or generic language (e.g. "contact us with questions") does NOT count
   as adequately addressing a specific GDPR right or requirement.
5. Keep your answer concise: a verdict (Compliant / Partial / Gap), then
   2-4 sentences of justification with citations.
"""


def generate_answer(question: str, context_block: str, client: "genai.Client"):
    user_message = f"""Context:
{context_block}

Question: {question}"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        config={"system_instruction": SYSTEM_PROMPT},
        contents=user_message,
    )
    return response.text


def main():
    if len(sys.argv) < 2:
        question = "Does our policy explain how customers can request deletion of their data?"
        print(f"No question provided, using default:\n\"{question}\"\n")
    else:
        question = " ".join(sys.argv[1:])

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        print("Get a free key at https://aistudio.google.com/apikey")
        print("Then set it with: $env:GOOGLE_API_KEY=\"your-key-here\"  (PowerShell)")
        sys.exit(1)

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    client_db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client_db.get_collection(COLLECTION_NAME)

    print("Retrieving relevant chunks...")
    results = retrieve(question, embed_model, collection)
    context_block = build_context_block(results)

    print("Asking Gemini...\n")
    gemini_client = genai.Client(api_key=api_key)
    answer = generate_answer(question, context_block, gemini_client)

    print(f"Question: {question}\n")
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()