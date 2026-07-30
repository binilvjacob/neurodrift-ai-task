import re

from pinecone import Pinecone, ServerlessSpec

from app.config import TENANT_ID_PATTERN
from app.vectorstore.base import VectorMatch, VectorRecord, VectorStore

_TENANT_ID_RE = re.compile(TENANT_ID_PATTERN)
_UPSERT_BATCH_SIZE = 100


class PineconeVectorStore(VectorStore):
    """One Pinecone serverless index, one namespace per tenant_id. Never accepts a namespace
    that hasn't been validated against TENANT_ID_PATTERN, even if a caller upstream already did.
    """

    def __init__(self, api_key: str, index_name: str, cloud: str, region: str, dimension: int) -> None:
        self._client = Pinecone(api_key=api_key)
        existing = {index.name for index in self._client.list_indexes()}
        if index_name not in existing:
            self._client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        self._index = self._client.Index(index_name)

    @staticmethod
    def _validate_tenant_id(tenant_id: str) -> None:
        if not _TENANT_ID_RE.match(tenant_id):
            raise ValueError(f"invalid tenant_id {tenant_id!r}: must match {TENANT_ID_PATTERN}")

    def upsert(self, tenant_id: str, records: list[VectorRecord]) -> None:
        self._validate_tenant_id(tenant_id)
        for i in range(0, len(records), _UPSERT_BATCH_SIZE):
            batch = records[i : i + _UPSERT_BATCH_SIZE]
            self._index.upsert(
                vectors=[{"id": r.id, "values": r.values, "metadata": r.metadata} for r in batch],
                namespace=tenant_id,
            )

    def query(self, tenant_id: str, vector: list[float], top_k: int) -> list[VectorMatch]:
        self._validate_tenant_id(tenant_id)
        result = self._index.query(vector=vector, top_k=top_k, namespace=tenant_id, include_metadata=True)
        return [
            VectorMatch(id=match["id"], score=match["score"], metadata=match.get("metadata") or {})
            for match in result["matches"]
        ]

    def delete_document(self, tenant_id: str, document_id: str) -> None:
        self._validate_tenant_id(tenant_id)
        prefix = f"{document_id}:"
        ids_to_delete: list[str] = []
        for id_batch in self._index.list(prefix=prefix, namespace=tenant_id):
            ids_to_delete.extend(id_batch)
        if ids_to_delete:
            self._index.delete(ids=ids_to_delete, namespace=tenant_id)
