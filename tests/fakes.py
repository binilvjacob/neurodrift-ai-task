import hashlib
import math
import re

from app.config import TENANT_ID_PATTERN
from app.embeddings.base import EmbeddingProvider
from app.vectorstore.base import VectorMatch, VectorRecord, VectorStore

_TENANT_ID_RE = re.compile(TENANT_ID_PATTERN)


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic bag-of-words hash embedding. No model, no network — just enough
    signal (shared tokens raise cosine similarity) to exercise retrieval routing without
    depending on a real embedding model.
    """

    _DIM = 32

    @property
    def dimension(self) -> int:
        return self._DIM

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._DIM
        for token in text.lower().split():
            index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self._DIM
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


class FakeVectorStore(VectorStore):
    """In-memory VectorStore keyed by tenant_id, one dict per namespace — mirrors
    Pinecone's namespace-per-tenant model closely enough that a query issued under one
    tenant_id can structurally never see another tenant's vectors, regardless of score.
    """

    def __init__(self) -> None:
        self._namespaces: dict[str, dict[str, VectorRecord]] = {}

    @staticmethod
    def _validate_tenant_id(tenant_id: str) -> None:
        if not _TENANT_ID_RE.match(tenant_id):
            raise ValueError(f"invalid tenant_id {tenant_id!r}: must match {TENANT_ID_PATTERN}")

    def upsert(self, tenant_id: str, records: list[VectorRecord]) -> None:
        self._validate_tenant_id(tenant_id)
        namespace = self._namespaces.setdefault(tenant_id, {})
        for record in records:
            namespace[record.id] = record

    def query(self, tenant_id: str, vector: list[float], top_k: int) -> list[VectorMatch]:
        self._validate_tenant_id(tenant_id)
        namespace = self._namespaces.get(tenant_id, {})
        scored = [
            VectorMatch(id=record.id, score=_cosine(vector, record.values), metadata=record.metadata)
            for record in namespace.values()
        ]
        scored.sort(key=lambda match: match.score, reverse=True)
        return scored[:top_k]

    def delete_document(self, tenant_id: str, document_id: str) -> None:
        self._validate_tenant_id(tenant_id)
        namespace = self._namespaces.get(tenant_id, {})
        prefix = f"{document_id}:"
        for key in [k for k in namespace if k.startswith(prefix)]:
            del namespace[key]
