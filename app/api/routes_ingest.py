import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi import Path as PathParam

from app.api.deps import get_ingestion_pipeline
from app.config import TENANT_ID_PATTERN
from app.ingestion.pipeline import IngestionPipeline
from app.models.schemas import IngestResponse

router = APIRouter()

_EXTENSION_TO_SOURCE_TYPE = {".pdf": "pdf", ".docx": "docx", ".txt": "txt", ".xlsx": "xlsx"}


@router.post("/tenants/{tenant_id}/documents", response_model=IngestResponse)
def upload_document(
    tenant_id: str = PathParam(pattern=TENANT_ID_PATTERN),
    file: UploadFile = File(...),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    suffix = Path(file.filename).suffix.lower()
    source_type = _EXTENSION_TO_SOURCE_TYPE.get(suffix)
    if source_type is None:
        raise HTTPException(status_code=400, detail=f"unsupported file type: {suffix or '(none)'}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = Path(tmp.name)

    try:
        return pipeline.ingest_file(tenant_id, tmp_path, file.filename, source_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
