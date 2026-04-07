from __future__ import annotations

from pathlib import Path
import re
import time

import pandas as pd
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

RUN_STATES = ["Idle", "Queued", "Processing", "Success", "Error"]


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
        margin-bottom: 1rem;
      }}
      .workspace-card {{
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 0.8rem;
        background: rgba(15, 23, 42, 0.72);
      }}
      .source-line {{
        border-left: 3px solid {THEME['accent_color']};
        padding-left: 0.6rem;
        margin: 0.35rem 0;
      }}
      .run-state-pill {{
        display: inline-block;
        border-radius: 999px;
        font-weight: 700;
        padding: 0.3rem 0.7rem;
        margin-bottom: 0.5rem;
      }}
      .run-state-idle {{ background:#334155; color:#E2E8F0; }}
      .run-state-queued {{ background:#A16207; color:#FEF3C7; }}
      .run-state-processing {{ background:#1D4ED8; color:#DBEAFE; }}
      .run-state-success {{ background:#166534; color:#DCFCE7; }}
      .run-state-error {{ background:#991B1B; color:#FEE2E2; }}
      mark {{
        background-color: rgba(14, 165, 233, 0.35);
        color: #E2E8F0;
        padding: 0 0.15rem;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _response_error(response: dict) -> str | None:
    error = response.get("error")
    if error:
        return str(response.get("detail") or error)
    return None


@st.cache_data(ttl=3600)
def ensure_sample_data_loaded(api_url: str, api_key: str) -> dict:
    """Auto-ingest bundled sample files if collection is empty."""
    headers = build_headers(api_key)
    health = api_call_with_retry(f"{api_url}/health", method="GET", headers=headers, timeout=20, max_retries=2)
    health_error = _response_error(health)
    if health_error:
        return {"ok": False, "message": health_error}

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

        result_error = _response_error(result)
        if result_error:
            return {
                "ok": False,
                "message": f"Failed to ingest {filename}: {result_error}",
            }

        ingested.append(filename)

    refreshed = api_call_with_retry(f"{api_url}/health", method="GET", headers=headers, timeout=20, max_retries=1)
    refreshed_error = _response_error(refreshed)
    return {
        "ok": refreshed_error is None,
        "ingested": ingested,
        "collection_count": int(refreshed.get("collection_count", 0)) if refreshed_error is None else 0,
        "message": refreshed_error or "",
    }


def _init_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("upload_history", [])
    st.session_state.setdefault("next_prompt", "")
    st.session_state.setdefault("show_sources", True)
    st.session_state.setdefault("top_k", 5)
    st.session_state.setdefault("bootstrap_result", None)
    st.session_state.setdefault("query_state", "Idle")
    st.session_state.setdefault("query_state_detail", "Ready for investigation query")
    st.session_state.setdefault("start_prompt_choice", SAMPLE_QUESTIONS[0])
    st.session_state.setdefault("start_prompt_text", "")
    st.session_state.setdefault("last_prompt", "")
    st.session_state.setdefault("last_query_error", "")
    st.session_state.setdefault("selected_source", None)
    st.session_state.setdefault("viewer_query", "")
    st.session_state.setdefault("viewer_sample_doc", "sample_fraud_report.txt")


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
        score = float(source.get("score", 0.0))
        page_text = f" (p.{page})" if page is not None else ""

        st.markdown(
            f"<div class='source-line'><strong>{source_name}{page_text}</strong> - {score:.2f}</div>",
            unsafe_allow_html=True,
        )


def _extract_source_snippet(source: dict) -> str:
    for key in ["snippet", "text", "content", "chunk_text", "quote"]:
        value = source.get(key)
        if value:
            return str(value).strip()
    return ""


def _render_source_preview(source: dict) -> None:
    source_name = str(source.get("source", "unknown"))
    page = source.get("page")
    score = float(source.get("score", 0.0))
    page_text = f" p.{page}" if page is not None else ""

    snippet = _extract_source_snippet(source)
    if len(snippet) > 160:
        snippet = f"{snippet[:157]}..."

    line = f"Evidence preview: {source_name}{page_text} · score {score:.2f}"
    if snippet:
        line = f"{line} · {snippet}"
    st.caption(line)


def _set_run_state(state: str, detail: str) -> None:
    if state not in RUN_STATES:
        state = "Error"
    st.session_state.query_state = state
    st.session_state.query_state_detail = detail


def _render_run_state() -> None:
    state = st.session_state.query_state
    css_class = state.lower()
    st.markdown(
        f"<span class='run-state-pill run-state-{css_class}'>{state}</span>"
        f" <span style='color:#94A3B8'>{st.session_state.query_state_detail}</span>",
        unsafe_allow_html=True,
    )


def _render_lifecycle(current_state: str) -> str:
    sequence = ["Queued", "Processing", "Success", "Error"]
    if current_state not in sequence:
        return ""

    active_idx = sequence.index(current_state)
    lines = ["#### Run Lifecycle"]
    for idx, name in enumerate(sequence):
        if current_state == "Error" and name == "Error":
            icon = "X"
        elif idx < active_idx and current_state != "Error":
            icon = "OK"
        elif idx == active_idx:
            icon = "..."
        else:
            icon = "-"
        lines.append(f"- {icon} {name}")
    return "\n".join(lines)


def _query_keywords(query: str) -> list[str]:
    words = [word.lower() for word in re.findall(r"[A-Za-z0-9_]+", query)]
    filtered = [word for word in words if len(word) >= 4]
    unique: list[str] = []
    for token in filtered:
        if token not in unique:
            unique.append(token)
    return unique[:8]


def _highlight_snippet(snippet: str, query: str) -> str:
    highlighted = snippet
    for token in _query_keywords(query):
        pattern = re.compile(rf"(?i)\\b({re.escape(token)})\\b")
        highlighted = pattern.sub(r"<mark>\\1</mark>", highlighted)
    return highlighted


def _default_source_for_viewer() -> tuple[dict | None, str]:
    selected = st.session_state.selected_source
    if selected:
        return selected, st.session_state.viewer_query

    for message in reversed(st.session_state.chat_history):
        if message.get("role") != "assistant":
            continue
        sources = message.get("sources") or []
        if sources:
            return sources[0], message.get("question", "")
    return None, ""


def _set_viewer_source(source: dict, query: str) -> None:
    st.session_state.selected_source = dict(source)
    st.session_state.viewer_query = query


_init_state()

if st.session_state.bootstrap_result is None:
    with st.spinner("Preparing sample corpus for the first visit..."):
        st.session_state.bootstrap_result = ensure_sample_data_loaded(API_URL, API_KEY)

st.markdown("<div class='main-header'>FraudShield - Analyst Workbench</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Split-screen investigation: evidence on the left, conversational analysis on the right.</div>",
    unsafe_allow_html=True,
)
_render_run_state()

bootstrap = st.session_state.bootstrap_result or {}
status_tab, guide_tab = st.sidebar.tabs(["Status", "Guide"])

with status_tab:
    st.header("System")
    api_connected, health = show_api_status(API_URL, headers=HEADERS)

    if health:
        st.metric("Indexed Chunks", int(health.get("collection_count", 0)))
        st.caption(f"Provider: {health.get('llm_provider', 'unknown')}")

    if bootstrap.get("ok") and bootstrap.get("ingested"):
        st.info(
            f"Auto-ingested: {', '.join(bootstrap['ingested'])} "
            f"(collection={bootstrap.get('collection_count', 0)})"
        )
    elif bootstrap and not bootstrap.get("ok"):
        st.warning(f"Sample auto-ingest skipped: {bootstrap.get('message', 'unknown error')}")

    if st.button("Clear Chat", key="clear_chat_btn", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.selected_source = None
        st.session_state.last_query_error = ""
        _set_run_state("Idle", "Ready for investigation query")
        st.rerun()

with guide_tab:
    st.markdown("### What This Project Is")
    st.markdown(
        "FraudShield is a retrieval-augmented investigation assistant. "
        "It searches indexed fraud documents and answers with source-grounded evidence."
    )
    st.markdown("### Process")
    st.markdown(
        "1. Documents are ingested and chunked.\n"
        "2. Chunks are embedded and stored in vector search.\n"
        "3. Your question retrieves top relevant chunks.\n"
        "4. The model generates an answer from retrieved context.\n"
        "5. Sources can be clicked and inspected in the document pane."
    )

with st.sidebar.expander("Workbench Drawer", expanded=True):
    st.markdown("#### Retrieval")
    st.slider("Top-K", min_value=1, max_value=20, key="top_k")
    st.toggle("Show full source list", key="show_sources")

    st.markdown("#### Upload")
    upload = st.file_uploader("Upload PDF / CSV / TXT", type=["pdf", "csv", "txt"], key="drawer_upload")
    source_type = st.selectbox("Source Type", options=["pdf", "csv", "text"], index=0, key="drawer_source_type")

    if st.button("Ingest Uploaded File", key="drawer_ingest_btn", use_container_width=True):
        if upload is None:
            st.error("Choose a file first.")
        else:
            with st.spinner("Ingesting file..."):
                response = api_call_with_retry(
                    f"{API_URL}/ingest",
                    method="POST",
                    headers=HEADERS,
                    files={"file": (upload.name, upload.getvalue(), upload.type or "application/octet-stream")},
                    data={"source_type": source_type},
                    timeout=120,
                    max_retries=2,
                )

            response_error = _response_error(response)
            if response_error:
                st.error(response_error)
            else:
                summary = (
                    f"{upload.name}: {response.get('chunks_created', 0)} chunks | "
                    f"collection={response.get('collection_size', 0)}"
                )
                st.session_state.upload_history.append(summary)
                st.success("Ingest successful")

    if st.session_state.upload_history:
        st.caption("Recent uploads")
        for item in st.session_state.upload_history[-5:][::-1]:
            st.write(f"- {item}")

st.markdown("### Start Here")
start_cols = st.columns([2, 2, 1])
with start_cols[0]:
    st.selectbox(
        "Quick prompt library",
        options=SAMPLE_QUESTIONS,
        key="start_prompt_choice",
        help="Choose a known investigation prompt for fast evaluation.",
    )
with start_cols[1]:
    st.text_input(
        "Or type your own question",
        key="start_prompt_text",
        placeholder="Example: Which providers show repeated near-threshold transfers?",
    )
with start_cols[2]:
    st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    if st.button("Run Prompt", key="start_here_run", use_container_width=True):
        chosen = st.session_state.start_prompt_text.strip() or st.session_state.start_prompt_choice
        st.session_state.next_prompt = chosen
        st.rerun()

if st.session_state.last_query_error and st.session_state.last_prompt:
    st.warning(f"Last query failed: {st.session_state.last_query_error}")
    if st.button("Retry Last Query", key="retry_last_query_btn", use_container_width=False):
        st.session_state.next_prompt = st.session_state.last_prompt
        st.rerun()

left_pane, right_pane = st.columns([1.05, 1.35])

with left_pane:
    st.markdown("### Document Viewer")
    viewer_source, viewer_query = _default_source_for_viewer()
    source_tab, corpus_tab = st.tabs(["Source Focus", "Corpus"])

    with source_tab:
        if viewer_source:
            source_name = str(viewer_source.get("source", "unknown"))
            page = viewer_source.get("page")
            score = float(viewer_source.get("score", 0.0))
            page_text = f" p.{page}" if page is not None else ""

            st.caption(f"Focused evidence: {source_name}{page_text} · score {score:.2f}")
            snippet = _extract_source_snippet(viewer_source)
            if snippet:
                highlighted = _highlight_snippet(snippet, viewer_query)
                st.markdown(highlighted, unsafe_allow_html=True)
            else:
                st.info("This source has no snippet payload. Expand full source list from chat cards.")

            metadata = viewer_source.get("metadata") or {}
            if metadata:
                with st.expander("Source Metadata", expanded=False):
                    st.json(metadata)
        else:
            st.info("Run a question and click a source chip in the chat pane to focus evidence here.")

    with corpus_tab:
        st.caption("Bundled sample corpus")
        sample_docs = [
            "sample_fraud_report.txt",
            "sample_transactions.csv",
        ]
        st.selectbox("Preview sample", options=sample_docs, key="viewer_sample_doc")
        sample_path = SAMPLE_DIR / st.session_state.viewer_sample_doc
        if sample_path.exists():
            if sample_path.suffix.lower() == ".csv":
                frame = pd.read_csv(sample_path)
                st.dataframe(frame.head(20), use_container_width=True, hide_index=True)
            else:
                st.text(sample_path.read_text(encoding="utf-8"))

        if st.session_state.upload_history:
            st.markdown("#### Recently Ingested")
            for item in st.session_state.upload_history[-5:][::-1]:
                st.write(f"- {item}")

with right_pane:
    st.markdown("### Investigation Chat")
    st.caption("Ask follow-up questions continuously. Click citations to focus the source in the left pane.")

    for message_idx, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("role") == "assistant":
                sources = message.get("sources") or []
                if sources:
                    _render_source_preview(sources[0])

                    chip_cols = st.columns(3)
                    for idx, source in enumerate(sources[:6]):
                        source_name = str(source.get("source", "unknown"))
                        page = source.get("page")
                        score = float(source.get("score", 0.0))
                        page_txt = f" p.{page}" if page is not None else ""
                        label = f"{source_name}{page_txt} ({score:.2f})"
                        with chip_cols[idx % 3]:
                            if st.button(label, key=f"cite_btn_{message_idx}_{idx}", use_container_width=True):
                                _set_viewer_source(source, str(message.get("question", "")))
                                st.rerun()

                    if st.session_state.show_sources:
                        with st.expander(f"Sources ({len(sources)})", expanded=False):
                            _render_sources(sources)

                st.caption(
                    f"Time {message.get('query_time_ms', '-')} ms"
                    f" | Tokens {message.get('tokens_used', '-') if message.get('tokens_used') is not None else '-'}"
                )

    with st.expander("More Sample Prompts", expanded=False):
        chip_cols = st.columns(2)
        for idx, question in enumerate(SAMPLE_QUESTIONS):
            with chip_cols[idx % 2]:
                if st.button(question, key=f"sample_q_{idx}", use_container_width=True):
                    st.session_state.next_prompt = question
                    st.rerun()

prompt = st.chat_input("Ask about fraud patterns...")
if not prompt and st.session_state.next_prompt:
    prompt = st.session_state.next_prompt
    st.session_state.next_prompt = ""

if prompt:
    st.session_state.last_prompt = prompt
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        state_box = st.empty()
        lifecycle_box = st.empty()

        _set_run_state("Queued", "Request accepted")
        state_box.info("Run State: Queued - Request accepted")
        lifecycle_box.markdown(_render_lifecycle("Queued"))
        time.sleep(0.1)

        _set_run_state("Processing", "Retrieving context and generating response")
        state_box.info("Run State: Processing - Retrieving context and generating response")
        lifecycle_box.markdown(_render_lifecycle("Processing"))

        result = api_call_with_retry(
            f"{API_URL}/query",
            method="POST",
            headers=HEADERS,
            json_payload={
                "question": prompt,
                "top_k": int(st.session_state.top_k),
                "include_sources": True,
            },
            timeout=120,
            max_retries=2,
        )

        result_error = _response_error(result)
        if result_error:
            st.session_state.last_query_error = result_error
            _set_run_state("Error", result_error)
            state_box.error(f"Run State: Error - {result_error}")
            lifecycle_box.markdown(_render_lifecycle("Error"))
            st.caption("Runtime Mode: API ERROR")
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": f"I could not complete that query: {result_error}",
                    "sources": [],
                    "query_time_ms": "-",
                    "tokens_used": None,
                    "question": prompt,
                }
            )
        else:
            st.session_state.last_query_error = ""

            answer = result.get("answer", "No answer returned.")
            _stream_text(answer)

            sources = result.get("sources", [])
            if sources:
                _set_viewer_source(sources[0], prompt)
                _render_source_preview(sources[0])
                if st.session_state.show_sources:
                    with st.expander(f"Sources ({len(sources)} documents)", expanded=False):
                        _render_sources(sources)

            _set_run_state("Success", "Answer generated successfully")
            state_box.success("Run State: Success - Answer generated successfully")
            lifecycle_box.markdown(_render_lifecycle("Success"))
            st.caption("Runtime Mode: LIVE API")

            st.caption(
                f"Time {result.get('query_time_ms', '-')} ms"
                f" | Tokens {result.get('tokens_used', '-') if result.get('tokens_used') is not None else '-'}"
            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "query_time_ms": result.get("query_time_ms", "-"),
                    "tokens_used": result.get("tokens_used"),
                    "question": prompt,
                }
            )

    st.rerun()

show_footer()
