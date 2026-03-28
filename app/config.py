from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    llm_provider: str = "openai"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "fraud_documents"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    chunk_size: int = 512
    chunk_overlap: int = 50

    top_k: int = 5
    rerank_top_k: int = 3
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    max_upload_size_mb: int = Field(default=50, ge=1, le=200)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
