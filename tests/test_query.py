def test_query_returns_answer_and_sources(client):
    ingest_response = client.post(
        "/ingest",
        files={
            "file": (
                "sample.csv",
                "transaction_id,account_id,amount\nTX-1,AC-1,9900\n",
                "text/csv",
            )
        },
        data={"source_type": "csv", "metadata": '{"category":"case"}'},
    )
    assert ingest_response.status_code == 200

    query_response = client.post(
        "/query",
        json={
            "question": "What suspicious pattern appears?",
            "top_k": 5,
            "include_sources": True,
        },
    )

    assert query_response.status_code == 200
    payload = query_response.json()
    assert "Synthetic answer" in payload["answer"]
    assert payload["tokens_used"] == 42
    assert payload["query_time_ms"] >= 0
    assert len(payload["sources"]) >= 1


def test_query_returns_404_when_collection_empty(client):
    response = client.post(
        "/query",
        json={
            "question": "Is there data?",
            "top_k": 5,
            "include_sources": True,
        },
    )

    assert response.status_code == 404
    assert "Ingest files first" in response.json()["detail"]


def test_query_supports_no_sources(client):
    ingest_response = client.post(
        "/ingest",
        files={"file": ("sample.txt", "simple fraud note", "text/plain")},
        data={"source_type": "text", "metadata": "{}"},
    )
    assert ingest_response.status_code == 200

    response = client.post(
        "/query",
        json={
            "question": "Summarize evidence",
            "include_sources": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_query_rate_limit_returns_429(client, fake_settings, fake_vector_store):
    fake_settings.api_key = "test-api-key"
    fake_settings.rate_limit_query_per_minute = 1
    fake_vector_store.records.append(
        {
            "content": "Fraud signal present",
            "metadata": {"source": "case.txt"},
            "score": 0.9,
            "vector": [0.1, 0.2, 0.3],
        }
    )

    headers = {
        "X-API-Key": "test-api-key",
        "X-Forwarded-For": "203.0.113.10",
    }
    first = client.post("/query", json={"question": "What happened?"}, headers=headers)
    second = client.post("/query", json={"question": "What happened?"}, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 429
