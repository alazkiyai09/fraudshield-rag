import json

import requests
import streamlit as st

API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="FraudShield RAG", page_icon="FS", layout="wide")
st.title("FraudShield RAG Agent")
st.caption("Upload fraud documents and run natural-language investigations with citations.")

left, right = st.columns(2)

with left:
    st.subheader("1) Ingest Document")
    upload = st.file_uploader("Upload PDF / CSV / TXT", type=["pdf", "csv", "txt"])
    source_type = st.selectbox("Source type", options=["pdf", "csv", "text"])
    metadata = st.text_area("Metadata JSON", value='{"category": "compliance", "year": 2025}')

    if st.button("Ingest"):
        if upload is None:
            st.error("Select a file first.")
        else:
            files = {"file": (upload.name, upload.getvalue(), upload.type or "application/octet-stream")}
            data = {"source_type": source_type, "metadata": metadata}
            try:
                response = requests.post(f"{API_BASE_URL}/ingest", files=files, data=data, timeout=120)
                if response.ok:
                    st.success("Ingestion successful")
                    st.json(response.json())
                else:
                    st.error(f"Ingestion failed ({response.status_code})")
                    st.json(response.json())
            except requests.RequestException as exc:
                st.error(f"API connection failed: {exc}")

with right:
    st.subheader("2) Query Fraud Corpus")
    question = st.text_area(
        "Question",
        value="What mule-account patterns appear in Q3 2025 transactions?",
        height=120,
    )
    top_k = st.slider("Top-K", min_value=1, max_value=20, value=5)
    include_sources = st.checkbox("Include sources", value=True)
    filters_text = st.text_input("Filters JSON (optional)", value="")

    if st.button("Run Query"):
        filters = None
        if filters_text.strip():
            try:
                filters = json.loads(filters_text)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid filter JSON: {exc.msg}")
                st.stop()

        payload = {
            "question": question,
            "top_k": top_k,
            "include_sources": include_sources,
            "filters": filters,
        }

        try:
            response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=120)
            if response.ok:
                data = response.json()
                st.markdown("### Answer")
                st.write(data.get("answer", ""))
                st.caption(f"Query time: {data.get('query_time_ms')} ms | Tokens: {data.get('tokens_used')}")

                if include_sources:
                    st.markdown("### Sources")
                    for idx, source in enumerate(data.get("sources", []), start=1):
                        st.markdown(f"**{idx}. {source.get('source')}** (score: {source.get('score'):.3f})")
                        st.write(source.get("content", ""))
            else:
                st.error(f"Query failed ({response.status_code})")
                st.json(response.json())
        except requests.RequestException as exc:
            st.error(f"API connection failed: {exc}")
