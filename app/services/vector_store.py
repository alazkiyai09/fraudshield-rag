from uuid import uuid4

from app.config import Settings
from app.services.document_loader import DocumentChunk


class VectorStoreError(RuntimeError):
    pass


class FraudVectorStore:
    """Qdrant wrapper for FraudShield document chunks."""

    def __init__(self, settings: Settings, vector_size: int) -> None:
        self.settings = settings
        self.vector_size = vector_size
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError("qdrant-client is not installed.") from exc

        try:
            self._client = QdrantClient(host=self.settings.qdrant_host, port=self.settings.qdrant_port)
            return self._client
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(f"Unable to initialize Qdrant client: {exc}") from exc

    def ensure_collection(self) -> None:
        client = self._ensure_client()

        try:
            from qdrant_client.http import models as qdrant_models
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError("qdrant-client models unavailable.") from exc

        collection_name = self.settings.qdrant_collection
        try:
            has_collection = client.collection_exists(collection_name)
        except Exception:
            has_collection = False

        if not has_collection:
            try:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.vector_size,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                )
            except Exception as exc:
                raise VectorStoreError(f"Failed creating collection '{collection_name}': {exc}") from exc

        # Optional indexes for common filters.
        for field_name, schema in (
            ("source", qdrant_models.PayloadSchemaType.KEYWORD),
            ("category", qdrant_models.PayloadSchemaType.KEYWORD),
            ("year", qdrant_models.PayloadSchemaType.INTEGER),
        ):
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception:
                continue

    def upsert_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise VectorStoreError("Mismatch between chunk count and vector count.")

        if not chunks:
            return 0

        client = self._ensure_client()
        self.ensure_collection()

        try:
            from qdrant_client.http import models as qdrant_models
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError("qdrant-client models unavailable.") from exc

        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            payload = {"content": chunk.content, **chunk.metadata}
            points.append(qdrant_models.PointStruct(id=str(uuid4()), vector=vector, payload=payload))

        try:
            client.upsert(
                collection_name=self.settings.qdrant_collection,
                wait=True,
                points=points,
            )
            return len(points)
        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert points into Qdrant: {exc}") from exc

    def search(self, query_vector: list[float], top_k: int, filters: dict | None = None) -> list[dict]:
        client = self._ensure_client()
        self.ensure_collection()

        qdrant_filter = self._build_filter(filters)

        try:
            results = client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed vector search: {exc}") from exc

        formatted_results: list[dict] = []
        for result in results:
            payload = result.payload or {}
            content = str(payload.get("content", ""))
            metadata = {key: value for key, value in payload.items() if key != "content"}

            formatted_results.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "score": float(result.score),
                    "id": str(result.id),
                }
            )

        return formatted_results

    def count(self) -> int:
        client = self._ensure_client()
        self.ensure_collection()

        try:
            count_result = client.count(collection_name=self.settings.qdrant_collection, exact=True)
            return int(count_result.count)
        except Exception as exc:
            raise VectorStoreError(f"Unable to count collection: {exc}") from exc

    def is_connected(self) -> bool:
        try:
            client = self._ensure_client()
            client.get_collections()
            return True
        except Exception:
            return False

    def _build_filter(self, filters: dict | None):
        if not filters:
            return None

        try:
            from qdrant_client.http import models as qdrant_models
        except ImportError:
            return None

        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                for item in value:
                    conditions.append(
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchValue(value=item),
                        )
                    )
            else:
                conditions.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value),
                    )
                )

        if not conditions:
            return None
        return qdrant_models.Filter(must=conditions)
