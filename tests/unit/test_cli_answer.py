"""The answer command: grounded output, refusal exit code and validation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from ingestion_test_support import make_document_chunks, make_vector_store

import rag_agent.__main__ as cli
from rag_agent.providers.base import ModelAuthError
from rag_agent.providers.fake import FakeChatModel, FakeEmbeddingModel

CONTENTS = ("主刷卡住时先断电并清理主刷", "充电座应放在干燥通风处")


@contextmanager
def fake_answering(*contents: str, reply: str) -> Iterator[tuple[Any, Any, FakeChatModel]]:
    vectors = make_vector_store()
    embedding_model = FakeEmbeddingModel(dimension=16)
    if contents:
        chunks = make_document_chunks("doc-test", *contents)
        vectors.upsert(chunks, embedding_model.embed([chunk.content for chunk in chunks]).vectors)
    yield vectors, embedding_model, FakeChatModel([reply])


def install(monkeypatch: pytest.MonkeyPatch, *contents: str, reply: str) -> None:
    monkeypatch.setattr(
        cli, "_open_answering", lambda settings: fake_answering(*contents, reply=reply)
    )


def answered_reply() -> str:
    return json.dumps(
        {
            "status": "answered",
            "answer": "先断电并清理主刷。[1]",
            "citations": [1],
            "reason": "",
        },
        ensure_ascii=False,
    )


def test_answer_requires_a_question() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["answer"])

    assert excinfo.value.code == 2


def test_answer_reports_citations(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch, *CONTENTS, reply=answered_reply())

    exit_code = cli.main(["answer", CONTENTS[0]])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["answer_status"] == "answered"
    assert payload["answer"] == "先断电并清理主刷。[1]"
    assert payload["citations"][0]["citation_id"] == 1
    assert payload["evidence"]


def test_answer_reports_a_refusal_with_exit_code_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reply = json.dumps(
        {"status": "insufficient", "answer": "", "citations": [], "reason": "资料未说明。"},
        ensure_ascii=False,
    )
    install(monkeypatch, *CONTENTS, reply=reply)

    exit_code = cli.main(["answer", "保修多久"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "refused"
    assert payload["answer_status"] == "refused"
    assert payload["refusal_cause"] == "model_insufficient"
    assert payload["reason"] == "资料未说明。"


def test_answer_refuses_when_the_index_is_empty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch, reply=answered_reply())

    exit_code = cli.main(["answer", "任何问题"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "refused"
    assert payload["citations"] == []
    assert payload["evidence"] == []


def test_answer_streams_raw_output_before_the_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch, *CONTENTS, reply=answered_reply())

    exit_code = cli.main(["answer", CONTENTS[0], "--stream"])
    captured = capsys.readouterr().out
    streamed, _, result = captured.partition("\n{")

    assert exit_code == 0
    assert streamed  # 真实的流式片段先于 JSON 输出
    assert answered_reply() in streamed
    payload = json.loads("{" + result)
    assert payload["status"] == "ok"
    assert payload["citations"][0]["citation_id"] == 1


def test_answer_streams_nothing_when_there_is_no_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch, reply=answered_reply())

    exit_code = cli.main(["answer", "任何问题", "--stream"])
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert captured.lstrip().startswith("{")
    assert json.loads(captured)["refusal_cause"] == "no_hits"


def test_answer_reports_missing_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    @contextmanager
    def failing(settings: Any) -> Iterator[tuple[Any, ...]]:
        raise ModelAuthError("no API key configured")
        yield  # pragma: no cover

    monkeypatch.setattr(cli, "_open_answering", failing)

    exit_code = cli.main(["answer", "问题"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["code"] == "auth"
