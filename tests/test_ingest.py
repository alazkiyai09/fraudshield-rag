import json

from app.dependencies import get_embedding_service
from app.main import app
from app.routers.ingest import _sanitize_filename


def test_ingest_csv_success(client):
    csv_content = (
        "transaction_id,account_id,amount,risk_flag\n"
        "TX-1,AC-1,9500,high\n"
        "TX-2,AC-2,150,low\n"
    )

    response = client.post(
        "/ingest",
        files={"file": ("sample.csv", csv_content, "text/csv")},
        data={
            "source_type": "csv",
            "metadata": json.dumps({"category": "compliance", "year": 2025}),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["documents_processed"] == 1
    assert payload["chunks_created"] >= 1
    assert payload["collection_size"] == payload["chunks_created"]


def test_ingest_rejects_invalid_metadata(client):
    response = client.post(
        "/ingest",
        files={"file": ("sample.csv", "a,b\n1,2", "text/csv")},
        data={
            "source_type": "csv",
            "metadata": "not-json",
        },
    )

    assert response.status_code == 400
    assert "metadata" in response.json()["detail"]


def test_ingest_rejects_unsupported_source_type(client):
    response = client.post(
        "/ingest",
        files={"file": ("sample.doc", "fake", "application/octet-stream")},
        data={
            "source_type": "doc",
            "metadata": "{}",
        },
    )

    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_ingest_hides_internal_embedding_errors(client):
    class BrokenEmbeddingService:
        def embed_documents(self, texts):
            raise RuntimeError("backend exploded")

    app.dependency_overrides[get_embedding_service] = lambda: BrokenEmbeddingService()
    try:
        response = client.post(
            "/ingest",
            files={"file": ("sample.txt", "simple fraud note", "text/plain")},
            data={"source_type": "text", "metadata": "{}"},
        )
    finally:
        app.dependency_overrides.pop(get_embedding_service, None)

    assert response.status_code == 500
    assert response.json()["detail"] == "Embedding generation failed."


def test_sanitize_filename_strips_paths_and_unsafe_characters():
    assert _sanitize_filename("../evil<script>.csv") == "evil_script_.csv"
