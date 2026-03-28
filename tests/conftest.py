from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.dependencies import (
    get_document_loader,
    get_embedding_service,
    get_rag_chain,
    get_retriever,
    get_vector_store,
)
from app.main import app
from app.services.document_loader import DocumentLoaderService


class FakeEmbeddingService:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), 0.1, 0.2] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [0.42, 0.11, 0.98]


class FakeVectorStore:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def upsert_chunks(self, chunks, vectors) -> int:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.records.append(
                {
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "score": 0.8,
                    "vector": vector,
                }
            )
        return len(chunks)

    def search(self, query_vector, top_k: int, filters: dict | None = None) -> list[dict]:
        results = self.records
        if filters:
            filtered = []
            for row in results:
                metadata = row.get("metadata", {})
                if all(metadata.get(key) == value for key, value in filters.items()):
                    filtered.append(row)
            results = filtered
        return results[:top_k]

    def count(self) -> int:
        return len(self.records)

    def is_connected(self) -> bool:
        return True


class FakeRetriever:
    def __init__(self, vector_store: FakeVectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(self, question: str, top_k: int | None = None, filters: dict | None = None) -> list[dict]:
        return self.vector_store.search(query_vector=[0.0, 0.0, 0.0], top_k=top_k or 5, filters=filters)


class FakeRAGChain:
    def generate_answer(self, question: str, chunks: list[dict]) -> tuple[str, int]:
        return (f"Synthetic answer for: {question}", 42)


@pytest.fixture()
def fake_settings() -> Settings:
    return Settings(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_collection="fraud_documents",
        embedding_dimension=3,
        chunk_size=120,
        chunk_overlap=10,
        top_k=5,
        rerank_top_k=3,
    )


@pytest.fixture()
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture()
def client(fake_settings: Settings, fake_vector_store: FakeVectorStore) -> Iterator[TestClient]:
    loader = DocumentLoaderService(fake_settings)
    embedding_service = FakeEmbeddingService()
    retriever = FakeRetriever(fake_vector_store)
    rag_chain = FakeRAGChain()

    app.dependency_overrides[get_settings] = lambda: fake_settings
    app.dependency_overrides[get_document_loader] = lambda: loader
    app.dependency_overrides[get_embedding_service] = lambda: embedding_service
    app.dependency_overrides[get_vector_store] = lambda: fake_vector_store
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_rag_chain] = lambda: rag_chain

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
