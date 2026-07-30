from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.deps import get_ingestion_pipeline, get_rag_pipeline
from app.api.routes_health import router as health_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_query import router as query_router
from app.logging_config import configure_logging

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly build the embedder, chunker/tokenizer, Pinecone client, and LLM client so
    # misconfiguration (bad credentials, an oversized chunk_size_tokens) fails at boot
    # instead of on the first upload or query request.
    get_ingestion_pipeline()
    get_rag_pipeline()
    yield


app = FastAPI(title="Multi-tenant RAG Pipeline", lifespan=lifespan)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)
