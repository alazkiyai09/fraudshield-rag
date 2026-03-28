from app.config import Settings


class EmbeddingServiceUnavailable(RuntimeError):
    pass


class EmbeddingService:
    """Wrapper around sentence-transformers embeddings."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingServiceUnavailable(
                "sentence-transformers is not installed; embeddings unavailable."
            ) from exc

        try:
            self._model = SentenceTransformer(self.settings.embedding_model)
            return self._model
        except Exception as exc:  # pragma: no cover
            raise EmbeddingServiceUnavailable(f"Failed to load embedding model: {exc}") from exc

    @property
    def embedding_dimension(self) -> int:
        if self._model is not None and hasattr(self._model, "get_sentence_embedding_dimension"):
            return int(self._model.get_sentence_embedding_dimension())
        return self.settings.embedding_dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._ensure_model()
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        vector = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return vector.tolist()
