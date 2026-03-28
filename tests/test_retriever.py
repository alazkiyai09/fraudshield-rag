import pytest
import sys
import types

from app.config import Settings
from app.services.retriever import CrossEncoderReranker, RetrieverService


class DummyEmbeddingService:
    def embed_query(self, question: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class DummyVectorStore:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.last_filters = None

    def search(self, query_vector, top_k: int, filters: dict | None = None) -> list[dict]:
        self.last_filters = filters
        return self.rows[:top_k]


class DummyReranker:
    def rerank(self, question: str, candidates: list[dict], top_k: int) -> list[dict]:
        ordered = sorted(candidates, key=lambda row: row["score"], reverse=True)
        return ordered[:top_k]


def test_retriever_applies_rerank_limit():
    rows = [
        {"content": "a", "metadata": {"source": "1"}, "score": 0.10},
        {"content": "b", "metadata": {"source": "2"}, "score": 0.90},
        {"content": "c", "metadata": {"source": "3"}, "score": 0.60},
    ]

    settings = Settings(top_k=5, rerank_top_k=2)
    service = RetrieverService(
        settings=settings,
        embedding_service=DummyEmbeddingService(),
        vector_store=DummyVectorStore(rows),
        reranker=DummyReranker(),
    )

    results = service.retrieve(question="find mule patterns", top_k=3)

    assert len(results) == 2
    assert results[0]["content"] == "b"
    assert results[1]["content"] == "c"


def test_retriever_passes_filters_to_vector_store():
    rows = [{"content": "a", "metadata": {"category": "compliance"}, "score": 0.5}]
    store = DummyVectorStore(rows)
    settings = Settings(top_k=5, rerank_top_k=3)

    service = RetrieverService(
        settings=settings,
        embedding_service=DummyEmbeddingService(),
        vector_store=store,
        reranker=DummyReranker(),
    )

    service.retrieve(question="filter test", filters={"category": "compliance"})

    assert store.last_filters == {"category": "compliance"}


def test_retriever_rejects_blank_question():
    settings = Settings(top_k=5, rerank_top_k=3)
    service = RetrieverService(
        settings=settings,
        embedding_service=DummyEmbeddingService(),
        vector_store=DummyVectorStore([]),
        reranker=DummyReranker(),
    )

    with pytest.raises(ValueError):
        service.retrieve(question="   ")


def test_retriever_returns_empty_when_no_hits():
    settings = Settings(top_k=5, rerank_top_k=3)
    service = RetrieverService(
        settings=settings,
        embedding_service=DummyEmbeddingService(),
        vector_store=DummyVectorStore([]),
        reranker=DummyReranker(),
    )

    assert service.retrieve(question="nothing to find") == []


def test_cross_encoder_reranker_with_mocked_model(monkeypatch):
    class FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def predict(self, pairs):
            assert len(pairs) == 2
            return [0.2, 0.9]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=FakeCrossEncoder))

    reranker = CrossEncoderReranker("mock-model")
    candidates = [
        {"content": "candidate-a", "score": 0.5},
        {"content": "candidate-b", "score": 0.4},
    ]
    results = reranker.rerank("query", candidates, top_k=1)

    assert len(results) == 1
    assert results[0]["content"] == "candidate-b"
    assert "rerank_score" in results[0]


def test_cross_encoder_reranker_falls_back_to_vector_score(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace())

    reranker = CrossEncoderReranker("mock-model")
    candidates = [
        {"content": "low", "score": 0.1},
        {"content": "high", "score": 0.9},
    ]
    results = reranker.rerank("query", candidates, top_k=1)

    assert len(results) == 1
    assert results[0]["content"] == "high"
