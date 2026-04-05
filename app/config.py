from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    llm_provider: str = "openai"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    llm_model: str = "gpt-4o-mini"

    api_key: str = ""

    qdrant_mode: str = "local"  # memory | local | network
    qdrant_url: str = ""
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_https: bool = False
    qdrant_path: str = "data/qdrant"
    qdrant_api_key: str = ""
    qdrant_collection: str = "fraud_documents"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    chunk_size: int = 512
    chunk_overlap: int = 50

    top_k: int = 5
    rerank_top_k: int = 3
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    max_upload_size_mb: int = Field(default=50, ge=1, le=200)
    rate_limit_ingest_per_minute: int = Field(default=10, ge=0, le=10000)
    rate_limit_query_per_minute: int = Field(default=30, ge=0, le=10000)
    cors_allow_origins: str = (
        "http://localhost:3000,"
        "http://127.0.0.1:3000,"
        "http://localhost:8501,"
        "http://127.0.0.1:8501,"
        "http://localhost:7860,"
        "http://127.0.0.1:7860"
    )

    def parsed_cors_allow_origins(self) -> list[str]:
        origins = [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]
        return origins or ["http://localhost:3000"]

    def normalized_qdrant_mode(self) -> str:
        return self.qdrant_mode.strip().lower()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
