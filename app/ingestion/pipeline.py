import hashlib
import logging
import time
from pathlib import Path

from app.embeddings.base import EmbeddingProvider
from app.ingestion.chunker import Chunker
from app.ingestion.loaders import extract_segments
from app.models.schemas import DocumentMetadata, IngestResponse
from app.vectorstore.base import VectorRecord, VectorStore

logger = logging.getLogger("rag.ingest")


def compute_document_id(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


class IngestionPipeline:
    def __init__(self, chunker: Chunker, embedder: EmbeddingProvider, vector_store: VectorStore) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    def ingest_file(self, tenant_id: str, file_path: Path, filename: str, source_type: str) -> IngestResponse:
        start = time.perf_counter()
        content = file_path.read_bytes()
        document_id = compute_document_id(content)
        metadata = DocumentMetadata(
            tenant_id=tenant_id,
            document_id=document_id,
            filename=filename,
            source_type=source_type,
        )

        segments = extract_segments(file_path, source_type)
        chunks = self._chunker.split(segments, metadata)

        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        records = [
            VectorRecord(
                id=chunk.chunk_id,
                values=vector,
                metadata={
                    "document_id": chunk.document_id,
                    "filename": filename,
                    "text": chunk.text,
                    # Pinecone metadata values must be flat (str/num/bool/list-of-str); the
                    # structured Locator is serialized to JSON and parsed back on retrieval.
                    "locator_json": chunk.locator.model_dump_json(exclude_none=True),
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        # document_id and chunk_id are both content-derived, so re-ingesting the same file
        # upserts the same vector ids with the same values: idempotent by construction.
        self._vector_store.upsert(tenant_id, records)

        logger.info(
            "document ingested",
            extra={
                "tenant_id": tenant_id,
                "document_id": document_id,
                # "filename" collides with a built-in LogRecord attribute (the source file
                # of the log call) and logging.Logger raises if `extra` tries to overwrite it.
                "doc_filename": filename,
                "chunks_indexed": len(chunks),
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return IngestResponse(
            tenant_id=tenant_id,
            document_id=document_id,
            filename=filename,
            chunks_indexed=len(chunks),
        )
