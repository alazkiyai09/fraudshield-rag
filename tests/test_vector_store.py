import sys
import types

import pytest

from app.config import Settings
from app.services.document_loader import DocumentChunk
from app.services.vector_store import FraudVectorStore, VectorStoreError


class FakeResult:
    def __init__(self, idx: int, payload: dict, score: float) -> None:
        self.id = idx
        self.payload = payload
        self.score = score


class FakeClientState:
    def __init__(self) -> None:
        self.collections = set()
        self.points = []
        self.payload_indexes = []
        self.init_kwargs = {}


class FakeQdrantClient:
    def __init__(self, **kwargs) -> None:
        self.state = GLOBAL_STATE
        self.state.init_kwargs = kwargs

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.state.collections

    def create_collection(self, collection_name: str, vectors_config):
        self.state.collections.add(collection_name)

    def create_payload_index(self, collection_name: str, field_name: str, field_schema):
        self.state.payload_indexes.append((collection_name, field_name, field_schema))

    def upsert(self, collection_name: str, wait: bool, points: list):
        self.state.points.extend(points)

    def search(self, collection_name: str, query_vector, query_filter, limit: int, with_payload: bool):
        rows = []
        for idx, point in enumerate(self.state.points, start=1):
            rows.append(FakeResult(idx=idx, payload=point.payload, score=0.9 - (idx * 0.1)))
        return rows[:limit]

    def query_points(self, collection_name: str, query, query_filter, limit: int, with_payload: bool):
        return types.SimpleNamespace(
            points=self.search(
                collection_name=collection_name,
                query_vector=query,
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
            )
        )

    def count(self, collection_name: str, exact: bool):
        return types.SimpleNamespace(count=len(self.state.points))

    def get_collections(self):
        return {"collections": list(self.state.collections)}


class FakeVectorParams:
    def __init__(self, size: int, distance):
        self.size = size
        self.distance = distance


class FakePointStruct:
    def __init__(self, id: str, vector: list[float], payload: dict):
        self.id = id
        self.vector = vector
        self.payload = payload


class FakeFieldCondition:
    def __init__(self, key: str, match):
        self.key = key
        self.match = match


class FakeMatchValue:
    def __init__(self, value):
        self.value = value


class FakeFilter:
    def __init__(self, must: list):
        self.must = must


GLOBAL_STATE = FakeClientState()


def install_fake_qdrant_modules(monkeypatch):
    fake_models = types.SimpleNamespace(
        VectorParams=FakeVectorParams,
        PointStruct=FakePointStruct,
        FieldCondition=FakeFieldCondition,
        MatchValue=FakeMatchValue,
        Filter=FakeFilter,
        Distance=types.SimpleNamespace(COSINE="cosine"),
        PayloadSchemaType=types.SimpleNamespace(KEYWORD="keyword", INTEGER="integer"),
    )

    monkeypatch.setitem(sys.modules, "qdrant_client", types.SimpleNamespace(QdrantClient=FakeQdrantClient))
    monkeypatch.setitem(sys.modules, "qdrant_client.http", types.SimpleNamespace(models=fake_models))
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", fake_models)


def test_vector_store_upsert_search_and_count(monkeypatch):
    GLOBAL_STATE.collections.clear()
    GLOBAL_STATE.points.clear()
    GLOBAL_STATE.payload_indexes.clear()
    install_fake_qdrant_modules(monkeypatch)

    settings = Settings(qdrant_collection="fraud_documents")
    store = FraudVectorStore(settings=settings, vector_size=3)

    chunks = [
        DocumentChunk(content="first", metadata={"source": "a.pdf", "year": 2025}),
        DocumentChunk(content="second", metadata={"source": "b.pdf", "category": "case"}),
    ]
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    upserted = store.upsert_chunks(chunks=chunks, vectors=vectors)
    results = store.search(query_vector=[0.1, 0.2, 0.3], top_k=2)

    assert upserted == 2
    assert store.count() == 2
    assert store.is_connected() is True
    assert len(results) == 2
    assert results[0]["metadata"]["source"] == "a.pdf"


def test_vector_store_build_filter(monkeypatch):
    install_fake_qdrant_modules(monkeypatch)
    store = FraudVectorStore(settings=Settings(), vector_size=3)

    built = store._build_filter({"category": "case", "year": 2025})
    assert isinstance(built, FakeFilter)
    assert len(built.must) == 2


def test_vector_store_rejects_mismatched_upsert_lengths(monkeypatch):
    install_fake_qdrant_modules(monkeypatch)
    store = FraudVectorStore(settings=Settings(), vector_size=3)

    with pytest.raises(VectorStoreError):
        store.upsert_chunks(
            chunks=[DocumentChunk(content="x", metadata={})],
            vectors=[],
        )


def test_vector_store_initializes_local_mode(monkeypatch):
    GLOBAL_STATE.init_kwargs = {}
    install_fake_qdrant_modules(monkeypatch)

    settings = Settings(qdrant_mode="local", qdrant_path="data/test-qdrant")
    store = FraudVectorStore(settings=settings, vector_size=3)
    store._ensure_client()

    assert GLOBAL_STATE.init_kwargs == {"path": "data/test-qdrant"}
