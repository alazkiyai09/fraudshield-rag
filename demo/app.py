from __future__ import annotations

from pathlib import Path
import time

import streamlit as st

from components.shared_components import (
    api_call_with_retry,
    build_headers,
    get_setting,
    show_api_status,
    show_footer,
)
from components.shared_theme import THEME

API_URL = get_setting("API_URL", "https://fraudshield-api-5tphgb6fsa-as.a.run.app").rstrip("/")
API_KEY = get_setting("API_KEY", "")
HEADERS = build_headers(API_KEY)
SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"

SAMPLE_QUESTIONS = [
    "What are the most common fraud patterns detected?",
    "Which transactions exceeded the reporting threshold?",
    "Summarize the mule account investigation findings",
    "What anomalies were detected in wire transfers?",
    "List the highest-risk providers and explain why",
]


st.set_page_config(page_title="FraudShield Demo", page_icon="🔍", layout="wide")

st.markdown(
    f"""
    <style>
      .stApp {{
        background: linear-gradient(180deg, {THEME['background']} 0%, #111827 100%);
      }}
      .main-header {{
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
      }}
      .sub-header {{
        color: #94A3B8;
        margin-bottom: 1.2rem;
      }}
      .source-line {{
        border-left: 3px solid {THEME['accent_color']};
        padding-left: 0.6rem;
        margin: 0.35rem 0;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600)
def ensure_sample_data_loaded(api_url: str, api_key: str) -> dict:
    """Auto-ingest bundled sample files if collection is empty."""
    headers = build_headers(api_key)
    health = api_call_with_retry(f"{api_url}/health", method="GET", headers=headers, timeout=20, max_retries=2)
    if "error" in health:
        return {"ok": False, "message": health.get("detail", health["error"])}

    initial_count = int(health.get("collection_count", 0))
    if initial_count > 0:
        return {"ok": True, "ingested": [], "collection_count": initial_count}

    sample_files = [
        ("sample_transactions.csv", "csv"),
        ("sample_fraud_report.txt", "text"),
    ]

    ingested: list[str] = []
    for filename, source_type in sample_files:
        path = SAMPLE_DIR / filename
        if not path.exists():
            return {"ok": False, "message": f"Missing sample file: {filename}"}

        with path.open("rb") as handle:
            result = api_call_with_retry(
                f"{api_url}/ingest",
                method="POST",
                headers=headers,
                files={"file": (filename, handle, "application/octet-stream")},
                data={"source_type": source_type},
                timeout=120,
                max_retries=2,
            )

        if "error" in result:
            return {
                "ok": False,
                "message": f"Failed to ingest {filename}: {result.get('detail', result['error'])}",
            }

        ingested.append(filename)

    refreshed = api_call_with_retry(f"{api_url}/health", method="GET", headers=headers, timeout=20, max_retries=1)
    return {
        "ok": "error" not in refreshed,
        "ingested": ingested,
        "collection_count": int(refreshed.get("collection_count", 0)) if "error" not in refreshed else 0,
        "message": refreshed.get("detail") if "error" in refreshed else "",
    }


def _init_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("upload_history", [])
    st.session_state.setdefault("next_prompt", "")
    st.session_state.setdefault("show_sources", True)
    st.session_state.setdefault("top_k", 5)
    st.session_state.setdefault("bootstrap_result", None)


def _stream_text(text: str) -> None:
    holder = st.empty()
    rendered = ""
    for token in text.split(" "):
        rendered = f"{rendered} {token}".strip()
        holder.markdown(rendered)
        time.sleep(0.01)


def _render_sources(sources: list[dict]) -> None:
    for source in sources:
        source_name = source.get("source", "unknown")
        page = source.get("page")
        score = source.get("score", 0.0)
        page_text = f" (p.{page})" if page is not None else ""

        st.markdown(
            f"<div class='source-line'><strong>{source_name}{page_text}</strong> — {score:.2f}</div>",
            unsafe_allow_html=True,
        )


_init_state()

if st.session_state.bootstrap_result is None:
    with st.spinner("Preparing sample corpus for the first visit..."):
        st.session_state.bootstrap_result = ensure_sample_data_loaded(API_URL, API_KEY)

st.markdown("<div class='main-header'>🔍 FraudShield — Fraud Investigation Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Ask questions about banking fraud reports using AI.</div>",
    unsafe_allow_html=True,
)

st.sidebar.header("📊 Status")
api_connected, health = show_api_status(API_URL, headers=HEADERS)

if health:
    st.sidebar.metric("Indexed Chunks", int(health.get("collection_count", 0)))
    st.sidebar.caption(f"Provider: {health.get('llm_provider', 'unknown')}")

bootstrap = st.session_state.bootstrap_result or {}
if bootstrap.get("ok") and bootstrap.get("ingested"):
    st.sidebar.info(
        f"Auto-ingested: {', '.join(bootstrap['ingested'])} "
        f"(collection={bootstrap.get('collection_count', 0)})"
    )
elif bootstrap and not bootstrap.get("ok"):
    st.sidebar.warning(f"Sample auto-ingest skipped: {bootstrap.get('message', 'unknown error')}")

if st.sidebar.button("🧹 Clear Chat", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Upload Document")
upload = st.sidebar.file_uploader("Upload PDF / CSV / TXT", type=["pdf", "csv", "txt"])
source_type = st.sidebar.selectbox("Source Type", options=["pdf", "csv", "text"], index=0)

if st.sidebar.button("Ingest Uploaded File", use_container_width=True):
    if upload is None:
        st.sidebar.error("Choose a file first.")
    else:
        with st.sidebar.spinner("Ingesting file..."):
            response = api_call_with_retry(
                f"{API_URL}/ingest",
                method="POST",
                headers=HEADERS,
                files={"file": (upload.name, upload.getvalue(), upload.type or "application/octet-stream")},
                data={"source_type": source_type},
                timeout=120,
                max_retries=2,
            )

        if "error" in response:
            st.sidebar.error(response.get("detail", response["error"]))
        else:
            summary = (
                f"{upload.name}: {response.get('chunks_created', 0)} chunks | "
                f"collection={response.get('collection_size', 0)}"
            )
            st.session_state.upload_history.append(summary)
            st.sidebar.success("Ingest successful")

if st.session_state.upload_history:
    st.sidebar.caption("Recent uploads")
    for item in st.session_state.upload_history[-5:][::-1]:
        st.sidebar.write(f"• {item}")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
st.session_state.top_k = st.sidebar.slider("Top-K", min_value=1, max_value=20, value=st.session_state.top_k)
st.session_state.show_sources = st.sidebar.toggle("Show sources", value=st.session_state.show_sources)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("role") == "assistant" and st.session_state.show_sources and message.get("sources"):
            with st.expander(f"📎 Sources ({len(message['sources'])})", expanded=False):
                _render_sources(message["sources"])
        if message.get("role") == "assistant":
            st.caption(
                f"⏱️ {message.get('query_time_ms', '-')} ms"
                f" | 🪙 {message.get('tokens_used', '-') if message.get('tokens_used') is not None else '-'} tokens"
            )

st.markdown("#### Quick Questions")
chip_cols = st.columns(2)
for idx, question in enumerate(SAMPLE_QUESTIONS):
    with chip_cols[idx % 2]:
        if st.button(question, key=f"sample_q_{idx}", use_container_width=True):
            st.session_state.next_prompt = question
            st.rerun()

prompt = st.chat_input("💬 Ask about fraud patterns...")
if not prompt and st.session_state.next_prompt:
    prompt = st.session_state.next_prompt
    st.session_state.next_prompt = ""

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching relevant evidence and drafting answer..."):
            result = api_call_with_retry(
                f"{API_URL}/query",
                method="POST",
                headers=HEADERS,
                json_payload={
                    "question": prompt,
                    "top_k": st.session_state.top_k,
                    "include_sources": st.session_state.show_sources,
                },
                timeout=120,
                max_retries=2,
            )

        if "error" in result:
            message = result.get("detail", result["error"])
            st.error(message)
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": f"I could not complete that query: {message}",
                    "sources": [],
                    "query_time_ms": "-",
                    "tokens_used": None,
                }
            )
        else:
            answer = result.get("answer", "No answer returned.")
            _stream_text(answer)

            sources = result.get("sources", []) if st.session_state.show_sources else []
            if st.session_state.show_sources and sources:
                with st.expander(f"📎 Sources ({len(sources)} documents)", expanded=False):
                    _render_sources(sources)

            st.caption(
                f"⏱️ {result.get('query_time_ms', '-')} ms"
                f" | 🪙 {result.get('tokens_used', '-') if result.get('tokens_used') is not None else '-'} tokens"
            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "query_time_ms": result.get("query_time_ms", "-"),
                    "tokens_used": result.get("tokens_used"),
                }
            )

    st.rerun()

show_footer()
