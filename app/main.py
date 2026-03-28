from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health_router, ingest_router, query_router

app = FastAPI(
    title="FraudShield RAG Agent",
    version="0.1.0",
    description=(
        "Retrieval-Augmented Generation API for fraud investigation across "
        "transaction reports, compliance docs, and case files."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/")
def root() -> dict:
    return {
        "name": "FraudShield RAG Agent",
        "status": "running",
        "docs": "/docs",
    }
