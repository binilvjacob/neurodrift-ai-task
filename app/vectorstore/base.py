from abc import ABC, abstractmethod

from pydantic import BaseModel


class VectorRecord(BaseModel):
    id: str
    values: list[float]
    metadata: dict


class VectorMatch(BaseModel):
    id: str
    score: float
    metadata: dict


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, tenant_id: str, records: list[VectorRecord]) -> None: ...

    @abstractmethod
    def query(self, tenant_id: str, vector: list[float], top_k: int) -> list[VectorMatch]: ...

    @abstractmethod
    def delete_document(self, tenant_id: str, document_id: str) -> None: ...
