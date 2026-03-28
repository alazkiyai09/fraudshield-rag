from app.config import Settings
from app.services.embeddings import EmbeddingService
from app.services.vector_store import FraudVectorStore


class CrossEncoderReranker:
    """Optional cross-encoder reranker used after vector similarity retrieval."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            return self._model
        except Exception:
            return None

    def rerank(self, question: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []

        model = self._ensure_model()
        if model is None:
            return sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]

        pairs = [(question, candidate.get("content", "")) for candidate in candidates]
        try:
            scores = model.predict(pairs)
        except Exception:
            return sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]

        reranked: list[dict] = []
        for candidate, rerank_score in zip(candidates, scores, strict=True):
            enriched = dict(candidate)
            enriched["rerank_score"] = float(rerank_score)
            reranked.append(enriched)

        return sorted(reranked, key=lambda item: item.get("rerank_score", 0.0), reverse=True)[:top_k]


class RetrieverService:
    """Two-stage retrieval: vector similarity followed by optional reranking."""

    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        vector_store: FraudVectorStore,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.reranker = reranker or CrossEncoderReranker(settings.rerank_model)

    def retrieve(self, question: str, top_k: int | None = None, filters: dict | None = None) -> list[dict]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        stage1_top_k = top_k or self.settings.top_k
        query_vector = self.embedding_service.embed_query(question)
        stage1_hits = self.vector_store.search(query_vector=query_vector, top_k=stage1_top_k, filters=filters)

        if not stage1_hits:
            return []

        stage2_top_k = min(self.settings.rerank_top_k, len(stage1_hits), stage1_top_k)
        return self.reranker.rerank(question=question, candidates=stage1_hits, top_k=stage2_top_k)
