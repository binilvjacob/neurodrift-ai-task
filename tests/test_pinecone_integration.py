import time

import pytest

from app.config import get_settings
from app.embeddings.bge import BGEEmbeddingProvider
from app.ingestion.chunker import Chunker
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.retriever import Retriever
from app.vectorstore.pinecone_store import PineconeVectorStore

TENANT_A = "pytest-integration-a"
TENANT_B = "pytest-integration-b"

DOC_A_TEXT = (
    "Pets under 25 kg are welcome in designated pet friendly rooms for a nightly fee of "
    "20 dollars. Guests must notify the property at the time of booking about their pet."
)
DOC_B_TEXT = (
    "To connect to the guest wifi network, select GuestNet from the list of available "
    "networks and enter the password printed on your room key card."
)

# Pinecone serverless has a brief read-after-write consistency lag; give newly upserted
# vectors time to become queryable before the tests assert against them.
_UPSERT_SETTLE_SECONDS = 2


def _pinecone_credentials_available() -> bool:
    try:
        settings = get_settings()
    except Exception:
        return False
    return bool(settings.pinecone_api_key) and bool(settings.pinecone_index_name)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _pinecone_credentials_available(),
        reason="requires PINECONE_API_KEY and PINECONE_INDEX_NAME to be set",
    ),
]


@pytest.fixture(scope="module")
def embedder() -> BGEEmbeddingProvider:
    settings = get_settings()
    return BGEEmbeddingProvider(
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
        query_instruction=settings.embedding_query_instruction,
    )


@pytest.fixture(scope="module")
def chunker() -> Chunker:
    settings = get_settings()
    return Chunker(
        tokenizer_name=settings.embedding_model_name,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )


@pytest.fixture(scope="module")
def vector_store(embedder) -> PineconeVectorStore:
    settings = get_settings()
    return PineconeVectorStore(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        dimension=embedder.dimension,
    )


@pytest.fixture(scope="module")
def ingestion_pipeline(chunker, embedder, vector_store) -> IngestionPipeline:
    return IngestionPipeline(chunker=chunker, embedder=embedder, vector_store=vector_store)


@pytest.fixture(scope="module")
def retriever(embedder, vector_store) -> Retriever:
    settings = get_settings()
    return Retriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=settings.top_k,
        score_threshold=settings.similarity_threshold,
    )


@pytest.fixture(scope="module")
def seeded_tenants(ingestion_pipeline, vector_store, tmp_path_factory):
    """Ingests one document into each of two real Pinecone namespaces, and always tears
    both back down afterward regardless of test outcome, so repeated runs never leave
    orphaned test data in a shared index.
    """
    tmp_dir = tmp_path_factory.mktemp("integration_docs")
    doc_a = doc_b = None
    try:
        doc_a_path = tmp_dir / "policy.txt"
        doc_a_path.write_text(DOC_A_TEXT, encoding="utf-8")
        doc_a = ingestion_pipeline.ingest_file(TENANT_A, doc_a_path, "policy.txt", "txt")

        doc_b_path = tmp_dir / "guide.txt"
        doc_b_path.write_text(DOC_B_TEXT, encoding="utf-8")
        doc_b = ingestion_pipeline.ingest_file(TENANT_B, doc_b_path, "guide.txt", "txt")

        time.sleep(_UPSERT_SETTLE_SECONDS)
        yield doc_a, doc_b
    finally:
        if doc_a is not None:
            vector_store.delete_document(TENANT_A, doc_a.document_id)
        if doc_b is not None:
            vector_store.delete_document(TENANT_B, doc_b.document_id)


def test_tenant_retrieves_its_own_document(seeded_tenants, retriever):
    doc_a, _ = seeded_tenants
    result = retriever.retrieve(TENANT_A, "What is the pet policy and fee?")
    assert result.is_grounded
    assert {source.document_id for source in result.matches} == {doc_a.document_id}


def test_other_tenant_retrieves_its_own_document(seeded_tenants, retriever):
    _, doc_b = seeded_tenants
    result = retriever.retrieve(TENANT_B, "How do I connect to the wifi?")
    assert result.is_grounded
    assert {source.document_id for source in result.matches} == {doc_b.document_id}


def test_query_never_returns_another_tenants_document(seeded_tenants, retriever):
    doc_a, _ = seeded_tenants
    # Ask tenant B the question that matches tenant A's content. Tenant B's Pinecone
    # namespace contains no pet-policy content at all, so tenant A's document must never
    # appear in the results regardless of how the score threshold falls.
    result = retriever.retrieve(TENANT_B, "What is the pet policy and fee?")
    returned_document_ids = {source.document_id for source in result.matches}
    assert doc_a.document_id not in returned_document_ids
