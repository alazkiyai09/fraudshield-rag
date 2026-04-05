def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "FraudShield RAG Agent"
    assert payload["status"] == "running"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["qdrant_connected"] is True
    assert payload["collection_count"] == 0


def test_query_requires_api_key_when_configured(client, fake_settings, fake_vector_store):
    fake_settings.api_key = "test-api-key"
    fake_vector_store.records.append(
        {
            "content": "Fraud signal present",
            "metadata": {"source": "case.txt"},
            "score": 0.9,
            "vector": [0.1, 0.2, 0.3],
        }
    )

    unauthorized = client.post("/query", json={"question": "What happened?"})
    authorized = client.post(
        "/query",
        json={"question": "What happened?"},
        headers={"X-API-Key": "test-api-key"},
    )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
