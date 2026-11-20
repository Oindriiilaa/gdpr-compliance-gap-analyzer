"""
app.py
Step 7: Streamlit UI for the Compliance RAG project.

Two modes:
  1. "Ask a Question" — single compliance question, grounded answer with citations
  2. "Full Gap Analysis" — scans all 16 GDPR articles and shows a scorecard + report

Run with:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

import generate as gen_module
import gap_report as report_module

st.set_page_config(page_title="GDPR Compliance Checker", page_icon="🛡️", layout="wide")


# --- Cached resources (load once, reuse across interactions) ---
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(gen_module.EMBED_MODEL_NAME)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=str(gen_module.CHROMA_DIR))
    return client.get_collection(gen_module.COLLECTION_NAME)


@st.cache_resource
def load_gemini_client(api_key):
    return genai.Client(api_key=api_key)


@st.cache_data
def load_gdpr_articles():
    return report_module.load_gdpr_articles()


# --- Sidebar: API key + mode ---
st.sidebar.title("🛡️ Compliance RAG")
api_key = st.sidebar.text_input("Google API Key", type="password",
                                 help="Get a free key at aistudio.google.com/apikey")
mode = st.sidebar.radio("Mode", ["Ask a Question", "Full Gap Analysis"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "This tool compares a synthetic company privacy policy against key GDPR "
    "articles using a RAG pipeline (local embeddings + Gemini). Built as a "
    "demo project — not legal advice."
)

if not api_key:
    st.info("Enter your Google API key in the sidebar to get started. "
             "Get a free one at https://aistudio.google.com/apikey")
    st.stop()

embed_model = load_embed_model()
collection = load_collection()
gemini_client = load_gemini_client(api_key)


# --- Mode 1: Ask a Question ---
if mode == "Ask a Question":
    st.title("Ask a Compliance Question")
    st.caption("Ask whether the company policy addresses a specific GDPR concept.")

    example_questions = [
        "Does our policy explain how customers can request deletion of their data?",
        "Does our policy state a legal basis for sending marketing emails?",
        "Does our policy explain what happens if there is a data breach?",
        "Does our policy describe our data security measures adequately?",
    ]
    chosen_example = st.selectbox("Try an example, or type your own below:",
                                   [""] + example_questions)
    question = st.text_input("Your question:", value=chosen_example)

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving relevant chunks and asking Gemini..."):
            results = gen_module.retrieve(question, embed_model, collection)
            context_block = gen_module.build_context_block(results)
            answer = gen_module.generate_answer(question, context_block, gemini_client)

        st.markdown("### Answer")
        st.markdown(answer)

        with st.expander("See retrieved source chunks (what the LLM was shown)"):
            st.markdown("**GDPR Articles:**")
            for doc, meta in zip(results["regulation"]["documents"][0], results["regulation"]["metadatas"][0]):
                st.markdown(f"- **{meta['source_label']}**: {doc}")
            st.markdown("**Policy Sections:**")
            for doc, meta in zip(results["company_policy"]["documents"][0], results["company_policy"]["metadatas"][0]):
                st.markdown(f"- **{meta['source_label']}**: {doc}")


# --- Mode 2: Full Gap Analysis ---
else:
    st.title("Full GDPR Gap Analysis")
    st.caption("Scans all 16 tracked GDPR articles against the company policy.")

    if st.button("Run Full Scan", type="primary"):
        articles = load_gdpr_articles()
        progress_bar = st.progress(0, text="Starting scan...")
        results_placeholder = st.empty()
        results = []

        for i, article in enumerate(articles, 1):
            progress_bar.progress(i / len(articles),
                                   text=f"Scanning {article['article']} ({i}/{len(articles)})...")

            query = f"{article['title']}: {article['summary']}"
            policy_chunks = report_module.retrieve_policy_chunks(query, embed_model, collection)
            prompt = report_module.build_prompt(article, policy_chunks)

            response = report_module.call_gemini_with_retry(
                gemini_client,
                model=report_module.GEMINI_MODEL,
                config={"system_instruction": report_module.SYSTEM_PROMPT},
                contents=prompt,
            )
            parsed = report_module.parse_verdict_json(response.text)

            results.append({
                "Article": article["article"],
                "Title": article["title"],
                "Verdict": parsed["verdict"],
                "Justification": parsed["justification"],
            })
            time.sleep(8)  # match the pacing used in gap_report.py

        progress_bar.empty()

        # Scorecard
        counts = {"Compliant": 0, "Partial": 0, "Gap": 0, "Unknown": 0}
        for r in results:
            counts[r["Verdict"]] = counts.get(r["Verdict"], 0) + 1

        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Compliant", counts["Compliant"])
        col2.metric("⚠️ Partial", counts["Partial"])
        col3.metric("❌ Gap", counts["Gap"])

        st.markdown("### Detailed Results")
        verdict_color = {"Compliant": "🟢", "Partial": "🟡", "Gap": "🔴", "Unknown": "⚪"}
        for r in results:
            with st.expander(f"{verdict_color.get(r['Verdict'], '⚪')} {r['Article']} — {r['Title']}: **{r['Verdict']}**"):
                st.write(r["Justification"])

        st.session_state["last_gap_results"] = results

    if "last_gap_results" in st.session_state:
        import json
        report_json = json.dumps(st.session_state["last_gap_results"], indent=2)
        st.download_button("Download Report (JSON)", report_json,
                            file_name="gdpr_gap_report.json", mime="application/json")