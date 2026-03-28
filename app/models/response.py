from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    content: str
    source: str
    page: int | None = None
    score: float
    metadata: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query_time_ms: float
    tokens_used: int | None = None


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int
    collection_size: int


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    collection_count: int
    embedding_model: str
    llm_provider: str
