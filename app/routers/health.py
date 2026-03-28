from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_vector_store
from app.models.response import HealthResponse
from app.services.vector_store import FraudVectorStore, VectorStoreError

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(
    settings: Settings = Depends(get_settings),
    vector_store: FraudVectorStore = Depends(get_vector_store),
) -> HealthResponse:
    qdrant_connected = vector_store.is_connected()
    collection_count = 0

    if qdrant_connected:
        try:
            collection_count = vector_store.count()
        except VectorStoreError:
            qdrant_connected = False

    return HealthResponse(
        status="healthy" if qdrant_connected else "degraded",
        qdrant_connected=qdrant_connected,
        collection_count=collection_count,
        embedding_model=settings.embedding_model,
        llm_provider=settings.llm_provider,
    )
