"""The evaluate command: modes, dataset errors and report writing."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from ingestion_test_support import make_vector_store

import rag_agent.__main__ as cli
from rag_agent.domain.documents import DocumentChunk, build_chunk_id
from rag_agent.evaluation.dataset import EvalCase
from rag_agent.providers.base import ModelAuthError
from rag_agent.providers.fake import FakeChatModel, FakeEmbeddingModel

PAGE_27_TEXT = "产品型号 DBX23 主机额定输入 20V 2A"


def make_chunk(content: str, *, page: int, index: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=build_chunk_id("doc-1", index),
        document_id="doc-1",
        source="manual.pdf",
        content=content,
        index=index,
        char_range=(0, len(content)),
        page=page,
    )


def cases() -> tuple[EvalCase, ...]:
    return (
        EvalCase(
            case_id="a",
            question=PAGE_27_TEXT,
            category="产品参数",
            answerable=True,
            expected_pages=(27,),
        ),
        EvalCase(
            case_id="b",
            question="这台机器人明年会涨价吗",
            category="范围外",
            answerable=False,
            expected_pages=(),
        ),
    )


@contextmanager
def fake_retrieval() -> Iterator[tuple[Any, FakeEmbeddingModel]]:
    vectors = make_vector_store()
    model = FakeEmbeddingModel(dimension=16)
    chunks = (make_chunk(PAGE_27_TEXT, page=27, index=1),)
    vectors.upsert(chunks, model.embed([chunk.content for chunk in chunks]).vectors)
    yield vectors, model


@contextmanager
def fake_answering() -> Iterator[tuple[Any, FakeEmbeddingModel, FakeChatModel]]:
    vectors = make_vector_store()
    embedding_model = FakeEmbeddingModel(dimension=16)
    chunks = (make_chunk(PAGE_27_TEXT, page=27, index=1),)
    vectors.upsert(chunks, embedding_model.embed([chunk.content for chunk in chunks]).vectors)
    reply = json.dumps(
        {"status": "answered", "answer": "型号是 DBX23。[1]", "citations": [1], "reason": ""},
        ensure_ascii=False,
    )
    insufficient = json.dumps(
        {
            "status": "insufficient",
            "answer": "",
            "citations": [],
            "reason": "资料未涉及价格与定价政策。",
        },
        ensure_ascii=False,
    )
    yield vectors, embedding_model, FakeChatModel([reply, insufficient])


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Patch the dataset, the stores and the report writer for one test."""

    written: list[dict[str, Any]] = []
    monkeypatch.setattr(cli, "load_cases", lambda path: cases())
    monkeypatch.setattr(cli, "_open_retrieval", lambda settings: fake_retrieval())
    monkeypatch.setattr(cli, "_open_answering", lambda settings: fake_answering())

    def fake_write(report: Any, out_dir: Path) -> tuple[Path, Path]:
        written.append({"mode": report.mode, "out_dir": str(out_dir)})
        return Path("report.md"), Path("report.json")

    monkeypatch.setattr(cli, "_write_reports", fake_write)
    return written


def test_retrieval_mode_prints_a_summary(
    patched: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["evaluate"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["mode"] == "retrieval"
    assert payload["report_markdown"] == "report.md"
    assert payload["summary"]["recall_at_k"] == 1.0
    assert payload["summary"]["answer_rate"] is None
    assert patched[0]["mode"] == "retrieval"


def test_answer_mode_measures_refusals(
    patched: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["evaluate", "--mode", "answer"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["mode"] == "answer"
    assert payload["summary"]["answer_rate"] == 1.0
    assert payload["summary"]["refusal_accuracy"] == 1.0
    assert patched[0]["mode"] == "answer"


def test_report_directory_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_cases", lambda path: cases())
    monkeypatch.setattr(cli, "_open_retrieval", lambda settings: fake_retrieval())
    seen: list[str] = []

    def fake_write(report: Any, out_dir: Path) -> tuple[Path, Path]:
        seen.append(str(out_dir))
        return Path("a.md"), Path("a.json")

    monkeypatch.setattr(cli, "_write_reports", fake_write)

    cli.main(["evaluate", "--out", "custom/reports"])
    capsys.readouterr()

    assert seen == [str(Path("custom/reports"))]


def test_limit_keeps_only_the_first_cases(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_cases", lambda path: cases()[:1])
    monkeypatch.setattr(cli, "_open_retrieval", lambda settings: fake_retrieval())
    monkeypatch.setattr(
        cli, "_write_reports", lambda report, out_dir: (Path("a.md"), Path("a.json"))
    )

    cli.main(["evaluate", "--limit", "1"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["summary"]["total"] == 1


def test_non_positive_limit_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["evaluate", "--limit", "0"])

    assert excinfo.value.code == 2


def test_missing_dataset_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rag_agent.evaluation.dataset import EvaluationDatasetError

    def boom(path: Any) -> Any:
        raise EvaluationDatasetError("evaluation set does not exist: nowhere.jsonl")

    monkeypatch.setattr(cli, "load_cases", boom)

    exit_code = cli.main(["evaluate", "--dataset", "nowhere.jsonl"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["code"] == "evaluation_dataset_error"


def test_missing_credentials_are_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    @contextmanager
    def failing(settings: Any) -> Iterator[tuple[Any, ...]]:
        raise ModelAuthError("no API key configured")
        yield  # pragma: no cover

    monkeypatch.setattr(cli, "load_cases", lambda path: cases())
    monkeypatch.setattr(cli, "_open_retrieval", failing)

    exit_code = cli.main(["evaluate"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["code"] == "auth"


def test_written_reports_contain_both_formats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_cases", lambda path: cases())
    monkeypatch.setattr(cli, "_open_retrieval", lambda settings: fake_retrieval())

    exit_code = cli.main(["evaluate", "--out", str(tmp_path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    markdown = Path(payload["report_markdown"]).read_text(encoding="utf-8")
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    assert "Recall@K" in markdown
    assert report_json["mode"] == "retrieval"
