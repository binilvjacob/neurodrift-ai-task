from fastapi import APIRouter, Depends
from fastapi import Path as PathParam

from app.api.deps import get_rag_pipeline
from app.config import TENANT_ID_PATTERN
from app.models.schemas import QueryRequest, QueryResponse
from app.rag.pipeline import RAGPipeline

router = APIRouter()


@router.post("/tenants/{tenant_id}/query", response_model=QueryResponse)
def query_tenant(
    request: QueryRequest,
    tenant_id: str = PathParam(pattern=TENANT_ID_PATTERN),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    return pipeline.answer(tenant_id, request.question, request.top_k)
