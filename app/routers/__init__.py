from app.routers.health import router as health_router
from app.routers.ingest import router as ingest_router
from app.routers.query import router as query_router

__all__ = ["health_router", "ingest_router", "query_router"]
