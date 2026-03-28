from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request to ingest a document into the vector store."""

    source_type: str = Field(..., description="Type: 'pdf', 'csv', 'text'")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata tags")


class QueryRequest(BaseModel):
    """Natural language query to the RAG system."""

    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] | None = Field(default=None, description="Metadata filters")
    include_sources: bool = Field(default=True)
