"""The ``rag-agent ask`` command, verified without network access."""

from __future__ import annotations

import json

import httpx
import pytest
from model_test_support import ScriptedTransport, completion_body
from model_test_support import build_chat_model as build_scripted_chat_model

import rag_agent.__main__ as cli

ERROR_BODY = {"error": {"message": "invalid api key"}}


def _install_scripted_model(monkeypatch: pytest.MonkeyPatch, transport: ScriptedTransport) -> None:
    monkeypatch.setattr(cli, "build_chat_model", lambda: build_scripted_chat_model(transport))


def test_ask_prints_a_structured_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    transport = ScriptedTransport(
        httpx.Response(200, json=completion_body("先检查充电座是否通电。"))
    )
    _install_scripted_model(monkeypatch, transport)

    exit_code = cli.main(["ask", "充不进电怎么办"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["text"] == "先检查充电座是否通电。"
    assert payload["model"] == "qwen-plus"
    assert payload["attempts"] == 1
    assert isinstance(payload["latency_ms"], float)
    assert transport.call_count == 1


def test_ask_without_a_question_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["ask"])

    assert excinfo.value.code == 2
    assert "ask requires a question" in capsys.readouterr().err


def test_ask_reports_provider_errors_as_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    transport = ScriptedTransport(httpx.Response(401, json=ERROR_BODY))
    _install_scripted_model(monkeypatch, transport)

    exit_code = cli.main(["ask", "问题"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["code"] == "auth"
    assert "test-key-not-real" not in payload["message"]


def test_health_command_still_reports_ok(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["health"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "ok"
