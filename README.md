# GDPR Compliance Gap Analyzer (RAG-based)

A Retrieval-Augmented Generation (RAG) system that compares a company's
privacy policy against key GDPR articles and flags compliance gaps —
built as a demonstration of applied GenAI for risk, audit, and compliance
use cases.

> ⚠️ **Disclaimer:** This is a demo/learning project using a fictional
> company policy. It is **not** legal advice and is not a substitute for
> review by a qualified compliance professional or legal counsel.

---

## Why this project

Compliance and audit teams spend significant time manually cross-referencing
internal policies against regulations like GDPR — a slow, expensive, and
inconsistent process given how dense and frequently-updated these
regulations are. This project explores how a RAG-based GenAI pipeline can
assist (not replace) that process: retrieving the exact relevant regulation
and policy text, then generating a grounded, cited compliance verdict.

## What it does

- **Ask a Question mode:** ask a specific compliance question (e.g. *"Does
  our policy explain how customers can request deletion of their data?"*)
  and get a grounded answer citing the exact GDPR article and policy
  section involved.
- **Full Gap Analysis mode:** automatically scans all 16 tracked GDPR
  articles against the policy and produces a structured report — verdict
  (Compliant / Partial / Gap) + justification for each — with a summary
  scorecard.

## Architecture

```
GDPR articles (JSON)  ─┐
                        ├─► Chunking ─► Embeddings (local, free) ─► ChromaDB
Company policy (.txt) ─┘                                              │
                                                                        ▼
                        User question / automated article loop ─► Retrieval
                                                                        │
                                                                        ▼
                                                    Gemini (grounded generation,
                                                    forced citations, strict
                                                    verdict on vague language)
                                                                        │
                                                                        ▼
                                                    Answer / Structured Report
```

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2), local | Free, no API key needed for retrieval |
| Vector DB | ChromaDB (local, persistent) | Simple, no cloud setup required |
| LLM | Google Gemini (`gemini-3.5-flash-lite`) | Free tier, no cost for a demo-scale project |
| UI | Streamlit | Fast way to build an interactive demo |
| Language | Python | — |

## Project structure

```
compliance-rag/
├── data/
│   ├── gdpr/gdpr_key_articles.json      # 16 GDPR articles, plain-English summaries
│   └── policy/sample_privacy_policy.txt  # Synthetic company policy (intentional gaps)
├── src/
│   ├── chunking.py         # Step 2: splits both sources into citable chunks
│   ├── embed_and_store.py  # Step 3: embeds chunks, stores in ChromaDB
│   ├── retrieve.py         # Step 4: retrieval-only test script
│   ├── debug_ranking.py    # Diagnostic: full similarity ranking for a query
│   ├── generate.py         # Step 5: single-question RAG + generation
│   └── gap_report.py       # Step 6: automated multi-article gap report
├── app.py                  # Step 7: Streamlit UI (both modes)
├── reports/                 # Generated gap reports (.md / .json)
├── chroma_db/                # Persistent vector DB (generated, not committed)
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

# Get a free API key: https://aistudio.google.com/apikey
$env:GOOGLE_API_KEY="your-key-here"   # PowerShell
```

## Usage

**Build the knowledge base (run once):**
```bash
python src/chunking.py
python src/embed_and_store.py
```

**CLI — ask a single question:**
```bash
python src/generate.py "Does our policy explain how customers can request deletion of their data?"
```

**CLI — full automated report:**
```bash
python src/gap_report.py
```

**Interactive UI:**
```bash
streamlit run app.py
```

## Sample result

Running the full gap analysis against the included synthetic policy:

| Verdict | Count |
|---|---|
| Partial | 4 |
| Gap | 12 |

The policy was deliberately written with realistic gaps (no clear legal
basis for marketing emails, no breach notification process, vague security
language, etc.) to give the tool something real to detect.

## Design decisions worth noting

- **Retrieval is split by source type** (top-k from GDPR + top-k from
  policy, separately) rather than one combined top-k — this guarantees
  both sides of the comparison are always present, rather than risking
  one source dominating the results.
- **The generation prompt is deliberately strict**: it's instructed to
  treat vague, generic language (e.g. "contact us with questions") as
  *not* sufficient to satisfy a specific GDPR right — preventing the LLM
  from being overly lenient.
- **Structured JSON output** for the automated report, with defensive
  parsing (LLMs sometimes wrap JSON in code fences despite instructions
  not to).
- **Retry-with-backoff** around the LLM calls — necessary in practice,
  since the free tier's rate limit was hit during development and the
  pipeline needed to recover gracefully rather than crash mid-scan.

## Limitations

- Uses a small, local embedding model (MiniLM) for retrieval. During
  development, this model under-ranked a short but relevant policy
  section in favor of longer, topically-similar-but-less-relevant text —
  a known limitation of smaller embedding models on short, sparse text.
  Mitigated here by retrieving a wider top-k, but a larger/more
  specialized embedding model would improve this further.
- LLM verdicts showed minor non-determinism on borderline cases across
  repeated runs. A production system would benefit from a lower
  temperature setting or a multi-sample consensus approach for
  higher-stakes verdicts.
- This is a prototype using a synthetic policy and a curated subset of
  GDPR (16 of 99 articles). A production deployment would need to handle
  much messier real-world documents (scanned PDFs, inconsistent
  formatting) and would require human review before any compliance
  decision is finalized — this tool is designed to assist that review,
  not replace it.

## Possible future improvements

- Hybrid retrieval (keyword + semantic) to address the short-text ranking issue above
- Support for uploading arbitrary policy documents (PDF parsing)
- Multi-regulation support (SOX, ISO 27001, DPDP Act)
- Confidence scoring / multi-sample consensus on verdicts