import pytest

from app.config import get_settings
from app.ingestion.chunker import Chunker
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.retriever import Retriever
from tests.fakes import FakeEmbeddingProvider, FakeVectorStore

# Fake embeddings are bag-of-words hashes, not semantic — much less separable than real
# BGE vectors, so tests use a lower threshold than the production default (0.5).
FAKE_SCORE_THRESHOLD = 0.2


@pytest.fixture(scope="session")
def chunker() -> Chunker:
    settings = get_settings()
    return Chunker(
        tokenizer_name=settings.embedding_model_name,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )


@pytest.fixture(scope="session")
def fake_embedder() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def ingestion_pipeline(chunker, fake_embedder, fake_vector_store) -> IngestionPipeline:
    return IngestionPipeline(chunker=chunker, embedder=fake_embedder, vector_store=fake_vector_store)


@pytest.fixture
def retriever(fake_embedder, fake_vector_store) -> Retriever:
    return Retriever(
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        top_k=5,
        score_threshold=FAKE_SCORE_THRESHOLD,
    )
