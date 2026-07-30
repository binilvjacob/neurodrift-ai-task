from pydantic import BaseModel

from app.embeddings.base import EmbeddingProvider
from app.models.schemas import Locator, SourceChunk
from app.vectorstore.base import VectorStore


class RetrievalResult(BaseModel):
    matches: list[SourceChunk]
    is_grounded: bool


class Retriever:
    def __init__(self, embedder: EmbeddingProvider, vector_store: VectorStore, top_k: int, score_threshold: float) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k
        self._score_threshold = score_threshold

    def retrieve(self, tenant_id: str, question: str, top_k: int | None = None) -> RetrievalResult:
        query_vector = self._embedder.embed_query(question)
        raw_matches = self._vector_store.query(tenant_id, query_vector, top_k or self._top_k)

        if not raw_matches or raw_matches[0].score < self._score_threshold:
            return RetrievalResult(matches=[], is_grounded=False)

        sources = [
            SourceChunk(
                chunk_id=match.id,
                document_id=match.metadata["document_id"],
                filename=match.metadata["filename"],
                text=match.metadata["text"],
                score=match.score,
                locator=Locator.model_validate_json(match.metadata["locator_json"]).display(),
            )
            for match in raw_matches
            if match.score >= self._score_threshold
        ]
        return RetrievalResult(matches=sources, is_grounded=True)
