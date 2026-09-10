"""The chunk-report command, driven without network access."""

from __future__ import annotations

import json

import pytest
from ingestion_test_support import make_document

import rag_agent.__main__ as cli
from rag_agent.domain.errors import DocumentNotFoundError


def test_chunk_report_requires_a_path(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["chunk-report"])

    assert excinfo.value.code == 2
    assert "chunk-report requires a file path" in capsys.readouterr().err


def test_chunk_report_reports_ingestion_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(path: str) -> object:
        raise DocumentNotFoundError("file does not exist: missing.md", source="missing.md")

    monkeypatch.setattr(cli, "load_document", boom)

    exit_code = cli.main(["chunk-report", "missing.md"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["code"] == "document_not_found"


def test_chunk_report_rejects_a_zero_chunk_size(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["chunk-report", "manual.md", "--chunk-sizes", "0"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["code"] == "invalid_argument"


def test_chunk_report_rejects_non_numeric_sizes(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["chunk-report", "manual.md", "--chunk-sizes", "abc"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["code"] == "invalid_argument"


def test_chunk_report_compares_every_combination(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("langchain_text_splitters")
    document = make_document(["# 标题\n" + "正文内容。" * 60])
    monkeypatch.setattr(cli, "load_document", lambda path: document)

    exit_code = cli.main(
        ["chunk-report", "manual.md", "--chunk-sizes", "120,240", "--chunk-overlaps", "0,40"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["source"] == document.source
    assert payload["page_count"] == 1
    assert [(item["chunk_size"], item["chunk_overlap"]) for item in payload["experiments"]] == [
        (120, 0),
        (120, 40),
        (240, 0),
        (240, 40),
    ]
    assert all(item["chunk_count"] > 0 for item in payload["experiments"])
    assert all(item["histogram"] for item in payload["experiments"])


def test_chunk_report_uses_defaults_when_no_flags_are_given(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("langchain_text_splitters")
    monkeypatch.setattr(cli, "load_document", lambda path: make_document(["正文内容。" * 40]))

    exit_code = cli.main(["chunk-report", "manual.md"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(payload["experiments"]) == 1
    assert payload["experiments"][0]["chunk_size"] == 800
