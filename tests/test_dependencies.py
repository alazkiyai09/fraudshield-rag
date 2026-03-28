from app.config import get_settings
from app.dependencies import (
    get_document_loader,
    get_embedding_service,
    get_rag_chain,
    get_retriever,
    get_vector_store,
)


def test_dependency_factories_construct_services():
    get_settings.cache_clear()
    get_document_loader.cache_clear()
    get_embedding_service.cache_clear()
    get_vector_store.cache_clear()
    get_retriever.cache_clear()
    get_rag_chain.cache_clear()

    settings = get_settings()
    loader = get_document_loader()
    embedding = get_embedding_service()
    store = get_vector_store()
    retriever = get_retriever()
    chain = get_rag_chain()

    assert loader.settings == settings
    assert embedding.settings == settings
    assert store.settings == settings
    assert retriever.settings == settings
    assert chain.settings == settings
