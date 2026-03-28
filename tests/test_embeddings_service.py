import sys
import types

from app.config import Settings
from app.services.embeddings import EmbeddingService


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        if len(texts) == 1:
            return [FakeVector([0.1, 0.2, 0.3])]
        return [FakeVector([float(index), 0.2, 0.3]) for index, _ in enumerate(texts, start=1)]


def test_embeddings_encode_documents_and_query(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    service = EmbeddingService(Settings(embedding_model="fake-model", embedding_dimension=3))

    vectors = service.embed_documents(["a", "b"])
    query_vector = service.embed_query("question")

    assert len(vectors) == 2
    assert vectors[0] == [1.0, 0.2, 0.3]
    assert query_vector == [0.1, 0.2, 0.3]
    assert service.embedding_dimension == 3
