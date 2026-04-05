from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.security import require_api_key
from app.routers import health_router, ingest_router, query_router

settings = get_settings()
cors_allow_origins = settings.parsed_cors_allow_origins()

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
    allow_origins=cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(ingest_router, dependencies=[Depends(require_api_key)])
app.include_router(query_router, dependencies=[Depends(require_api_key)])


@app.get("/")
def root() -> dict:
    return {
        "name": "FraudShield RAG Agent",
        "status": "running",
        "docs": "/docs",
    }
