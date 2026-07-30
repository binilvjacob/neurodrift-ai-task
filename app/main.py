from fastapi import FastAPI

from app.api.routes_health import router as health_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_query import router as query_router

app = FastAPI(title="Multi-tenant RAG Pipeline")

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)
