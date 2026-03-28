from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_rag_chain, get_retriever, get_vector_store
from app.models.request import QueryRequest
from app.models.response import QueryResponse, SourceDocument
from app.services.chain import RAGChainService
from app.services.retriever import RetrieverService
from app.services.vector_store import FraudVectorStore, VectorStoreError

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_documents(
    payload: QueryRequest,
    retriever: RetrieverService = Depends(get_retriever),
    rag_chain: RAGChainService = Depends(get_rag_chain),
    vector_store: FraudVectorStore = Depends(get_vector_store),
) -> QueryResponse:
    started = perf_counter()

    try:
        if vector_store.count() == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No documents found in collection. Ingest files first.",
            )
    except VectorStoreError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    try:
        hits = retriever.retrieve(question=payload.question, top_k=payload.top_k, filters=payload.filters)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except VectorStoreError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    answer, tokens_used = rag_chain.generate_answer(question=payload.question, chunks=hits)

    response_sources: list[SourceDocument] = []
    if payload.include_sources:
        for hit in hits:
            metadata = hit.get("metadata", {})
            response_sources.append(
                SourceDocument(
                    content=hit.get("content", ""),
                    source=str(metadata.get("source", "unknown-source")),
                    page=metadata.get("page"),
                    score=float(hit.get("rerank_score", hit.get("score", 0.0))),
                    metadata=metadata,
                )
            )

    elapsed_ms = (perf_counter() - started) * 1000

    return QueryResponse(
        answer=answer,
        sources=response_sources,
        query_time_ms=round(elapsed_ms, 2),
        tokens_used=tokens_used,
    )
