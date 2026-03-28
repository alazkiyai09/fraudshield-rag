import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.config import Settings, get_settings
from app.dependencies import get_document_loader, get_embedding_service, get_vector_store
from app.models.response import IngestResponse
from app.services.document_loader import DocumentLoaderService
from app.services.embeddings import EmbeddingService, EmbeddingServiceUnavailable
from app.services.vector_store import FraudVectorStore, VectorStoreError

router = APIRouter(tags=["ingest"])

_ALLOWED_SOURCE_TYPES = {"pdf", "csv", "text"}


def _parse_metadata(metadata_text: str | None) -> dict:
    if metadata_text is None or not metadata_text.strip():
        return {}

    try:
        parsed = json.loads(metadata_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"metadata must be valid JSON: {exc.msg}",
        ) from exc

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="metadata must be a JSON object.",
        )

    return parsed


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    source_type: str = Form(...),
    metadata: str | None = Form(default=None),
    settings: Settings = Depends(get_settings),
    document_loader: DocumentLoaderService = Depends(get_document_loader),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: FraudVectorStore = Depends(get_vector_store),
) -> IngestResponse:
    normalized_source_type = source_type.strip().lower()
    if normalized_source_type not in _ALLOWED_SOURCE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported source_type '{source_type}'. Allowed: {_ALLOWED_SOURCE_TYPES}.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb}MB.",
        )

    parsed_metadata = _parse_metadata(metadata)

    try:
        chunks = document_loader.load_and_chunk(
            filename=file.filename or "uploaded_file",
            file_bytes=file_bytes,
            source_type=normalized_source_type,
            metadata=parsed_metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No readable content found in uploaded file.",
        )

    chunk_texts = [chunk.content for chunk in chunks]

    try:
        vectors = embedding_service.embed_documents(chunk_texts)
    except EmbeddingServiceUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    try:
        chunks_created = vector_store.upsert_chunks(chunks=chunks, vectors=vectors)
        collection_size = vector_store.count()
    except VectorStoreError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return IngestResponse(
        status="success",
        documents_processed=1,
        chunks_created=chunks_created,
        collection_size=collection_size,
    )
