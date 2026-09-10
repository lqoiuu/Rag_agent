"""The search command: ranked output, confidence exit code and validation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from ingestion_test_support import make_document_chunks, make_vector_store

import rag_agent.__main__ as cli
from rag_agent.providers.fake import FakeEmbeddingModel


@contextmanager
def seeded_retrieval(*contents: str) -> Iterator[tuple[Any, FakeEmbeddingModel]]:
    vectors = make_vector_store()
    model = FakeEmbeddingModel(dimension=16)
    if contents:
        chunks = make_document_chunks("doc-test", *contents)
        vectors.upsert(chunks, model.embed([chunk.content for chunk in chunks]).vectors)
    yield vectors, model


def install(monkeypatch: pytest.MonkeyPatch, *contents: str) -> None:
    monkeypatch.setattr(cli, "_open_retrieval", lambda settings: seeded_retrieval(*contents))


def test_search_requires_a_query() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["search"])

    assert excinfo.value.code == 2


def test_search_reports_ranked_hits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch, "主刷卡住时先断电检查", "滤网多久清洗一次")

    exit_code = cli.main(["search", "主刷卡住时先断电检查"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["confident"] is True
    assert payload["hits"][0]["content"] == "主刷卡住时先断电检查"
    assert payload["hits"][0]["rank"] == 1


def test_search_reports_low_confidence_with_exit_code_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch, "主刷卡住时先断电检查")

    exit_code = cli.main(["search", "完全无关的问题", "--threshold", "0.99"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["confident"] is False
    assert "below the threshold" in payload["reason"]


def test_search_reports_an_empty_index(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch)

    exit_code = cli.main(["search", "任何问题"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["hits"] == []
    assert payload["best_score"] is None
    assert payload["confident"] is False


def test_search_honours_the_top_k_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch, *[f"第{index}段内容" for index in range(5)])

    exit_code = cli.main(["search", "第1段内容", "--top-k", "2"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(payload["hits"]) == 2
    assert payload["top_k"] == 2


def test_search_rejects_an_out_of_range_threshold(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch, "内容")

    exit_code = cli.main(["search", "内容", "--threshold", "5"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["code"] == "invalid_argument"
