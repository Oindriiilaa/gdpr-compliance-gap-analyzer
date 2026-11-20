"""
gap_report.py
Step 6 of the Compliance RAG pipeline.

Instead of asking one question at a time, this loops through ALL GDPR
articles in the knowledge base, retrieves the most relevant policy
sections for each, and asks Gemini for a structured verdict
(Compliant / Partial / Gap) with justification.

Outputs:
  - Prints a summary scorecard to the console
  - Saves a full markdown report to reports/gdpr_gap_report.md
  - Saves the raw structured data to reports/gdpr_gap_report.json

Usage:
    python src/gap_report.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from google.genai import errors as genai_errors

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
GDPR_PATH = BASE_DIR / "data" / "gdpr" / "gdpr_key_articles.json"
REPORTS_DIR = BASE_DIR / "reports"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "compliance_chunks"
TOP_K_POLICY_CHUNKS = 4  # per article, keep it tight since we loop 16x
# Free tier on Google AI Studio as of this writing — Flash-Lite typically
# has a higher free-tier RPM than standard Flash, useful since this script
# makes 16 sequential calls. The retry-with-backoff below is a safety net
# regardless of your account's actual limit.
GEMINI_MODEL = "gemini-3.5-flash-lite"

SYSTEM_PROMPT = """You are a compliance analysis assistant reviewing a company
privacy policy against a single GDPR article at a time.

You will be given the text of ONE GDPR article, plus the company policy
sections most likely to be relevant to it.

Respond with ONLY a JSON object (no markdown fences, no extra text) in
exactly this shape:
{
  "verdict": "Compliant" | "Partial" | "Gap",
  "justification": "2-3 sentences explaining the verdict, citing the exact
    policy section(s) you're referencing (e.g. 'Policy Section 6')."
}

Rules:
- "Compliant": the policy sections clearly and specifically address the
  article's requirement.
- "Partial": the policy touches on the general topic but is vague, generic,
  or missing key specifics the article requires.
- "Gap": the policy sections provided do not address this requirement at all.
- Be strict: generic language like "contact us with questions" does NOT
  count as adequately addressing a specific right or process.
- Base your answer ONLY on the text provided — no outside GDPR knowledge.
"""


def load_gdpr_articles():
    with open(GDPR_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_policy_chunks(query: str, embed_model, collection, k=TOP_K_POLICY_CHUNKS):
    query_embedding = embed_model.encode([query]).tolist()
    res = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where={"source_type": "company_policy"},
    )
    return list(zip(res["documents"][0], res["metadatas"][0]))


def build_prompt(article, policy_chunks):
    article_text = f"{article['article']} – {article['title']}: {article['summary']}"
    policy_text = "\n\n".join(f"[{meta['source_label']}] {doc}" for doc, meta in policy_chunks)

    return f"""GDPR Article:
{article_text}

Relevant Company Policy Sections:
{policy_text}

Evaluate whether the policy sections adequately address this article's requirement."""


def parse_verdict_json(raw_text: str):
    """Gemini sometimes wraps JSON in ```json fences despite instructions — strip them."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"verdict": "Unknown", "justification": f"(Could not parse response: {raw_text[:200]})"}


def call_gemini_with_retry(client, model, config, contents, max_retries=5):
    """
    Free tier allows only 5 requests/minute for gemini-2.5-flash. Even with
    pacing between calls, occasional 429s can happen — retry with backoff
    instead of crashing the whole 16-article scan over one hiccup.
    """
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model, config=config, contents=contents
            )
        except genai_errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait_time = 25 * (attempt + 1)  # back off further each retry
                print(f"    Rate limited — waiting {wait_time}s before retry "
                      f"({attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting.")


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    client_db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client_db.get_collection(COLLECTION_NAME)

    gemini_client = genai.Client(api_key=api_key)
    articles = load_gdpr_articles()

    results = []
    print(f"Scanning {len(articles)} GDPR articles against the policy...\n")

    for i, article in enumerate(articles, 1):
        query = f"{article['title']}: {article['summary']}"
        policy_chunks = retrieve_policy_chunks(query, embed_model, collection)
        prompt = build_prompt(article, policy_chunks)

        response = call_gemini_with_retry(
            gemini_client,
            model=GEMINI_MODEL,
            config={"system_instruction": SYSTEM_PROMPT},
            contents=prompt,
        )
        parsed = parse_verdict_json(response.text)

        print(f"[{i}/{len(articles)}] {article['article']}: {parsed['verdict']}")
        results.append({
            "article": article["article"],
            "title": article["title"],
            "verdict": parsed["verdict"],
            "justification": parsed["justification"],
        })

        time.sleep(8)  # conservative pacing — actual free-tier RPM varies by
                        # account, and the retry logic above is the real safety net

    # --- Summary scorecard ---
    counts = {"Compliant": 0, "Partial": 0, "Gap": 0, "Unknown": 0}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    print("\n" + "=" * 50)
    print("SUMMARY SCORECARD")
    print("=" * 50)
    for verdict, count in counts.items():
        if count > 0:
            print(f"{verdict}: {count}")
    print("=" * 50)

    # --- Save outputs ---
    REPORTS_DIR.mkdir(exist_ok=True)

    with open(REPORTS_DIR / "gdpr_gap_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_lines = ["# GDPR Compliance Gap Report\n", "## Summary\n"]
    for verdict, count in counts.items():
        if count > 0:
            md_lines.append(f"- **{verdict}**: {count}")
    md_lines.append("\n## Details\n")
    for r in results:
        md_lines.append(f"### {r['article']} — {r['title']}")
        md_lines.append(f"**Verdict:** {r['verdict']}\n")
        md_lines.append(f"{r['justification']}\n")

    with open(REPORTS_DIR / "gdpr_gap_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nFull report saved to:")
    print(f"  {REPORTS_DIR / 'gdpr_gap_report.md'}")
    print(f"  {REPORTS_DIR / 'gdpr_gap_report.json'}")


if __name__ == "__main__":
    main()