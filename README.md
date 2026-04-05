# FraudShield RAG Agent

FraudShield is a retrieval-augmented fraud investigation service that ingests fraud documents (PDF/CSV/text), stores embeddings in Qdrant, and answers natural-language analyst questions with source citations.

## Live Demo

| Interface | URL |
|---|---|
| Interactive Demo | https://fraudshield-demo-5tphgb6fsa-as.a.run.app |
| API Documentation | https://fraudshield-api-5tphgb6fsa-as.a.run.app/docs |

## Features

- FastAPI API with `POST /ingest`, `POST /query`, and `GET /health`
- Document chunking with `RecursiveCharacterTextSplitter` (`512` chunk size, `50` overlap)
- Sentence-transformer embeddings (`all-MiniLM-L6-v2`)
- Qdrant vector retrieval with metadata filtering
- Two-stage retrieval (`top_k` vector search + cross-encoder rerank)
- LLM generation through OpenAI or Anthropic via LangChain
- Streamlit UI for upload/query workflow
- RAGAS evaluation scaffold
- Docker Compose stack (`qdrant` + `fraudshield-api`)

## Project Layout

```text
project-1-fraudshield-rag/
├── app/
├── data/
├── eval/
├── tests/
├── streamlit_app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements.lock.txt
└── .env.example
```

## Quickstart

1. Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
```

Flexible install without the lockfile:

```bash
pip install -r requirements.txt
```

2. Configure environment:

```bash
cp .env.example .env
# Fill OPENAI_API_KEY or ANTHROPIC_API_KEY if you want live LLM generation
```

3. Start Qdrant + API:

```bash
docker compose up --build
```

4. Open API docs:

- `http://localhost:8000/docs`

5. Optional Streamlit UI:

```bash
streamlit run streamlit_app.py
```

## API Examples

### Ingest

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/sample/sample_transactions.csv" \
  -F "source_type=csv" \
  -F 'metadata={"category":"transactions","year":2025}'
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What mule-account patterns appear in Q3 2025?",
    "top_k": 5,
    "include_sources": true,
    "filters": {"year": 2025}
  }'
```

## Testing

```bash
pytest --cov=app --cov-report=term-missing -v
```

## Reproducible Dependency Audit

```bash
pip-audit -r requirements.lock.txt --no-deps --disable-pip
```

To refresh all project lockfiles from the workspace root:

```bash
./scripts/dependency_locks.sh compile
./scripts/dependency_locks.sh audit
```

## RAGAS Evaluation

```bash
python3 eval/evaluate_rag.py --dataset eval/test_questions.json
```

## Notes

- Retrieval and reranking are production-oriented but conservative on failure: if external models are unavailable, the service degrades gracefully.
- Replace sample files in `data/sample/` with your own fraud corpus for realistic quality metrics.
