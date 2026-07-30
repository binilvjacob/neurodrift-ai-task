import pytest

HOTEL_A_POLICY = (
    "Pets under 25 kg are welcome in designated pet friendly rooms for a nightly fee of "
    "20 dollars. Guests must notify the property at the time of booking about their pet."
)
HOTEL_B_GUIDE = (
    "To connect to the guest wifi network, select GuestNet from the list of available "
    "networks and enter the password printed on your room key card."
)


def _ingest_text(ingestion_pipeline, tmp_path, tenant_id: str, filename: str, text: str):
    file_path = tmp_path / filename
    file_path.write_text(text, encoding="utf-8")
    return ingestion_pipeline.ingest_file(tenant_id, file_path, filename, "txt")


def test_tenant_retrieves_its_own_document(ingestion_pipeline, retriever, tmp_path):
    doc_a = _ingest_text(ingestion_pipeline, tmp_path, "hotel-a", "policy.txt", HOTEL_A_POLICY)
    _ingest_text(ingestion_pipeline, tmp_path, "hotel-b", "guide.txt", HOTEL_B_GUIDE)

    result = retriever.retrieve("hotel-a", "What is the pet policy and fee?")

    assert result.is_grounded
    assert {source.document_id for source in result.matches} == {doc_a.document_id}


def test_query_never_returns_another_tenants_document(ingestion_pipeline, retriever, tmp_path):
    doc_a = _ingest_text(ingestion_pipeline, tmp_path, "hotel-a", "policy.txt", HOTEL_A_POLICY)
    _ingest_text(ingestion_pipeline, tmp_path, "hotel-b", "guide.txt", HOTEL_B_GUIDE)

    # Ask hotel-b the question that matches hotel-a's content. Even if the fake embedder
    # scored this above threshold for hotel-b's own (unrelated) document, the namespace
    # boundary means hotel-a's vectors are never in the candidate pool to begin with.
    result = retriever.retrieve("hotel-b", "What is the pet policy and fee?")

    returned_document_ids = {source.document_id for source in result.matches}
    assert doc_a.document_id not in returned_document_ids


def test_fresh_tenant_sees_nothing_from_other_tenants(ingestion_pipeline, retriever, tmp_path):
    _ingest_text(ingestion_pipeline, tmp_path, "hotel-a", "policy.txt", HOTEL_A_POLICY)
    _ingest_text(ingestion_pipeline, tmp_path, "hotel-b", "guide.txt", HOTEL_B_GUIDE)

    result = retriever.retrieve("hotel-c", "What is the pet policy and fee?")

    assert result.matches == []
    assert not result.is_grounded


def test_invalid_tenant_id_rejected_before_reaching_a_namespace(ingestion_pipeline, tmp_path):
    with pytest.raises(ValueError):
        _ingest_text(ingestion_pipeline, tmp_path, "bad tenant!", "policy.txt", HOTEL_A_POLICY)
