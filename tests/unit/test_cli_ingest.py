"""Command wiring for ingest, reindex and delete-document."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from ingestion_test_support import make_vector_store

import rag_agent.__main__ as cli
from rag_agent.ingestion.pipeline import INDEXED, REMOVED, IngestResult
from rag_agent.providers.base import ModelAuthError
from rag_agent.providers.fake import FakeEmbeddingModel
from rag_agent.storage import MetadataStore
from rag_agent.vectorstore import ChunkVectorStore


@contextmanager
def fake_index() -> Iterator[tuple[MetadataStore, ChunkVectorStore, FakeEmbeddingModel]]:
    with MetadataStore(":memory:") as store:
        yield store, make_vector_store(), FakeEmbeddingModel()


@pytest.fixture
def fake_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_open_index", lambda settings: fake_index())


def test_ingest_requires_a_path() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["ingest"])

    assert excinfo.value.code == 2


def test_ingest_reports_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "ingest_path",
        lambda path, **kwargs: IngestResult(
            source="manual.md", status=INDEXED, document_id="doc-1", chunk_count=7, vector_count=7
        ),
    )

    exit_code = cli.main(["ingest", "manual.md"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["results"][0]["chunk_count"] == 7


def test_ingest_reports_partial_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "ingest_path",
        lambda path, **kwargs: IngestResult(
            source="broken.pdf",
            status="failed",
            error_code="empty_document",
            error_message="document contains no extractable text",
        ),
    )

    exit_code = cli.main(["ingest", "broken.pdf"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "partial"
    assert payload["results"][0]["error_code"] == "empty_document"


def test_ingest_reports_missing_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    @contextmanager
    def failing(settings: Any) -> Iterator[tuple[Any, ...]]:
        raise ModelAuthError("no API key configured; set RAG_AGENT_QWEN_API_KEY")
        yield  # pragma: no cover

    monkeypatch.setattr(cli, "_open_index", failing)

    exit_code = cli.main(["ingest", "manual.md"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["code"] == "auth"


def test_delete_document_requires_a_source() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["delete-document"])

    assert excinfo.value.code == 2


def test_delete_document_reports_unknown_source(
    fake_stores: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["delete-document", "missing.pdf"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "partial"
    assert payload["results"][0]["error_code"] == "document_not_found"


def test_delete_document_reports_removal(
    fake_stores: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "remove_document",
        lambda source, **kwargs: IngestResult(
            source=source, status=REMOVED, document_id="doc-1", vector_count=9, chunk_count=9
        ),
    )

    exit_code = cli.main(["delete-document", "manual.md"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["results"][0]["vector_count"] == 9


def test_reindex_reports_results(
    fake_stores: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "sync_index",
        lambda directory, **kwargs: (
            IngestResult(source="manual.pdf", status=INDEXED, document_id="doc-1", chunk_count=4),
            IngestResult(source="gone.pdf", status=REMOVED, document_id="doc-2", vector_count=3),
        ),
    )

    exit_code = cli.main(["reindex"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert [item["status"] for item in payload["results"]] == [INDEXED, REMOVED]
