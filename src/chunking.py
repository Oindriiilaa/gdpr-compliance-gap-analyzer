"""
chunking.py
Step 2 of the Compliance RAG pipeline.

Loads:
  - data/gdpr/gdpr_key_articles.json   (already naturally chunked per article)
  - data/policy/sample_privacy_policy.txt (needs splitting by numbered section)

Outputs:
  - data/processed/chunks.json
    A flat list of chunks, each tagged with metadata so we can cite the
    exact source (e.g. "GDPR Article 17" or "Policy Section 6") later
    when the LLM answers a question.
"""

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
GDPR_PATH = BASE_DIR / "data" / "gdpr" / "gdpr_key_articles.json"
POLICY_PATH = BASE_DIR / "data" / "policy" / "sample_privacy_policy.txt"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "chunks.json"


def chunk_gdpr(path: Path) -> list[dict]:
    """Each GDPR article is already a natural, self-contained chunk."""
    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    chunks = []
    for i, art in enumerate(articles):
        text = f"{art['article']} – {art['title']}: {art['summary']}"
        chunks.append({
            "id": f"gdpr_{i}",
            "source_type": "regulation",
            "source_label": art["article"],
            "title": art["title"],
            "text": text,
        })
    return chunks


def chunk_policy(path: Path) -> list[dict]:
    """
    Split the policy into chunks by numbered top-level section
    (e.g. "1. INTRODUCTION", "2. WHAT DATA WE COLLECT", ...).
    We deliberately stop before the "NOTE FOR PROJECT" block since
    that's our own annotation, not part of the policy itself.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Cut off everything from our own dev note onward
    raw = raw.split("---\nNOTE FOR PROJECT")[0]

    # Split on lines like "3. HOW WE USE YOUR DATA"
    pattern = re.compile(r"\n(\d+)\.\s+([A-Z][A-Z \-']+)\n")
    matches = list(pattern.finditer(raw))

    chunks = []
    for idx, match in enumerate(matches):
        section_num = match.group(1)
        section_title = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()

        chunks.append({
            "id": f"policy_{section_num}",
            "source_type": "company_policy",
            "source_label": f"Policy Section {section_num}",
            "title": section_title.title(),
            "text": f"Section {section_num} ({section_title.title()}): {body}",
        })
    return chunks


def main():
    gdpr_chunks = chunk_gdpr(GDPR_PATH)
    policy_chunks = chunk_policy(POLICY_PATH)
    all_chunks = gdpr_chunks + policy_chunks

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"GDPR chunks:   {len(gdpr_chunks)}")
    print(f"Policy chunks: {len(policy_chunks)}")
    print(f"Total chunks:  {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
