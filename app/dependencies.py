from functools import lru_cache

from app.config import Settings, get_settings
from app.services.chain import RAGChainService
from app.services.document_loader import DocumentLoaderService
from app.services.embeddings import EmbeddingService
from app.services.retriever import RetrieverService
from app.services.vector_store import FraudVectorStore


@lru_cache(maxsize=1)
def get_document_loader() -> DocumentLoaderService:
    return DocumentLoaderService(get_settings())


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_settings())


@lru_cache(maxsize=1)
def get_vector_store() -> FraudVectorStore:
    settings: Settings = get_settings()
    return FraudVectorStore(settings=settings, vector_size=settings.embedding_dimension)


@lru_cache(maxsize=1)
def get_retriever() -> RetrieverService:
    settings = get_settings()
    return RetrieverService(
        settings=settings,
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


@lru_cache(maxsize=1)
def get_rag_chain() -> RAGChainService:
    return RAGChainService(get_settings())
