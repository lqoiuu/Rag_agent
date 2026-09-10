"""Metadata store behaviour over SQLite."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_agent.storage import IngestionJob, MetadataStore, StoredDocument, utc_now


def make_document(**overrides: object) -> StoredDocument:
    values: dict[str, object] = {
        "document_id": "doc-abc",
        "source": "manual.pdf",
        "file_type": "pdf",
        "checksum": "a" * 64,
        "version": "a" * 12,
        "size_bytes": 1234,
        "page_count": 32,
        "chunk_count": 40,
        "embedding_model": "text-embedding-v4",
        "updated_at": utc_now(),
    }
    values.update(overrides)
    return StoredDocument(**values)  # type: ignore[arg-type]


def make_job(**overrides: object) -> IngestionJob:
    values: dict[str, object] = {
        "job_id": "job-1",
        "source": "manual.pdf",
        "status": "succeeded",
        "started_at": utc_now(),
        "finished_at": utc_now(),
        "document_id": "doc-abc",
        "chunk_count": 40,
    }
    values.update(overrides)
    return IngestionJob(**values)  # type: ignore[arg-type]


@pytest.fixture
def store() -> MetadataStore:
    with MetadataStore(":memory:") as opened:
        yield opened


def test_initialize_is_idempotent(store: MetadataStore) -> None:
    store.initialize()
    store.initialize()

    assert store.list_documents() == ()
    assert store.list_jobs() == ()


def test_document_round_trip(store: MetadataStore) -> None:
    document = make_document()

    store.save_document(document)

    assert store.get_document("manual.pdf") == document
    assert store.list_documents() == (document,)


def test_missing_document_returns_none(store: MetadataStore) -> None:
    assert store.get_document("nope.pdf") is None


def test_saving_the_same_source_updates_the_row_and_records_versions(
    store: MetadataStore,
) -> None:
    store.save_document(make_document(version="a" * 12, checksum="a" * 64, chunk_count=40))
    store.save_document(make_document(version="b" * 12, checksum="b" * 64, chunk_count=12))

    documents = store.list_documents()
    assert len(documents) == 1
    assert documents[0].version == "b" * 12
    assert documents[0].chunk_count == 12

    versions = store.list_versions("doc-abc")
    assert [item.version for item in versions] == ["a" * 12, "b" * 12]


def test_saving_the_same_version_twice_keeps_one_version_row(store: MetadataStore) -> None:
    store.save_document(make_document())
    store.save_document(make_document())

    assert len(store.list_versions("doc-abc")) == 1


def test_documents_are_listed_by_source(store: MetadataStore) -> None:
    store.save_document(make_document(source="b.pdf", document_id="doc-b"))
    store.save_document(make_document(source="a.pdf", document_id="doc-a"))

    assert [document.source for document in store.list_documents()] == ["a.pdf", "b.pdf"]


def test_delete_removes_document_and_versions(store: MetadataStore) -> None:
    document = make_document()
    store.save_document(document)

    removed = store.delete_document("manual.pdf")

    assert removed == document
    assert store.get_document("manual.pdf") is None
    assert store.list_versions("doc-abc") == ()
    assert store.delete_document("manual.pdf") is None


def test_latest_job_returns_the_newest_attempt(store: MetadataStore) -> None:
    store.record_job(make_job(job_id="job-old", status="failed", error_code="empty_document"))
    store.record_job(make_job(job_id="job-new"))

    latest = store.latest_job("manual.pdf")

    assert latest is not None
    assert latest.job_id == "job-new"
    assert latest.status == "succeeded"
    assert store.latest_job("other.pdf") is None


def test_jobs_are_listed_newest_first_with_a_limit(store: MetadataStore) -> None:
    for index in range(3):
        store.record_job(make_job(job_id=f"job-{index}"))

    jobs = store.list_jobs(limit=2)

    assert [job.job_id for job in jobs] == ["job-2", "job-1"]


def test_job_error_fields_round_trip(store: MetadataStore) -> None:
    job = make_job(
        job_id="job-failed",
        status="failed",
        document_id=None,
        chunk_count=0,
        error_code="empty_document",
        error_message="document contains no extractable text",
    )

    store.record_job(job)

    assert store.latest_job("manual.pdf") == job


def test_file_backed_store_survives_reopening(tmp_path: Path) -> None:
    database = tmp_path / "metadata.sqlite3"

    with MetadataStore(database) as first:
        first.save_document(make_document())

    with MetadataStore(database) as second:
        assert second.get_document("manual.pdf") is not None
