from pydantic import BaseModel

from app.embeddings.base import EmbeddingProvider
from app.models.schemas import Locator, SourceChunk
from app.vectorstore.base import VectorMatch, VectorStore


class RetrievalResult(BaseModel):
    matches: list[SourceChunk]
    is_grounded: bool


def passes_threshold(matches: list[SourceChunk], threshold: float) -> bool:
    """The gate: grounded iff there's a top match and it clears the threshold.

    A free function (not a method) so the eval harness's threshold sweep can re-apply this
    exact same gate against one cached set of raw matches per question, for many candidate
    thresholds, without re-embedding the question or re-querying the vector store each time.
    """
    return bool(matches) and matches[0].score >= threshold


class Retriever:
    def __init__(self, embedder: EmbeddingProvider, vector_store: VectorStore, top_k: int, score_threshold: float) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k
        self._score_threshold = score_threshold

    def search(self, tenant_id: str, question: str, top_k: int | None = None) -> list[SourceChunk]:
        """Raw ranked results, ungated — one embed call and one vector store query."""
        query_vector = self._embedder.embed_query(question)
        raw_matches = self._vector_store.query(tenant_id, query_vector, top_k or self._top_k)
        return [self._to_source_chunk(match) for match in raw_matches]

    def retrieve(self, tenant_id: str, question: str, top_k: int | None = None) -> RetrievalResult:
        matches = self.search(tenant_id, question, top_k)

        if not passes_threshold(matches, self._score_threshold):
            return RetrievalResult(matches=[], is_grounded=False)

        sources = [match for match in matches if match.score >= self._score_threshold]
        return RetrievalResult(matches=sources, is_grounded=True)

    @staticmethod
    def _to_source_chunk(match: VectorMatch) -> SourceChunk:
        return SourceChunk(
            chunk_id=match.id,
            document_id=match.metadata["document_id"],
            filename=match.metadata["filename"],
            text=match.metadata["text"],
            score=match.score,
            locator=Locator.model_validate_json(match.metadata["locator_json"]).display(),
        )
