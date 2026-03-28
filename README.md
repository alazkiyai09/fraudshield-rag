# FraudShield RAG Agent

RAG-powered fraud investigation service over banking/compliance documents using LangChain, Qdrant, and FastAPI.

## Live Demo (Deployment Placeholder)

- API base URL: `https://fraudshield-api-xxxxx-as.a.run.app`
- Swagger docs: `https://fraudshield-api-xxxxx-as.a.run.app/docs`
- Status: `pending deployment`

## Features

- `POST /ingest`, `POST /query`, `GET /health`
- PDF/CSV/text ingestion with chunking
- Qdrant vector retrieval + cross-encoder rerank
- OpenAI/Anthropic model support
- Streamlit UI (`streamlit_app.py`)
- RAGAS evaluation scaffold

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up --build
```

API docs: `http://localhost:8000/docs`

Streamlit UI:

```bash
streamlit run streamlit_app.py
```

## Tests

```bash
pytest --cov=app --cov-report=term-missing -v
```

## Deployment Prep

Deploy to Cloud Run using gcloud or your CI pipeline.

Update the live links above after deployment.
