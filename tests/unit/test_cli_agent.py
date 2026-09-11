"""The agent command: one graph turn with its trace and exit code."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from agent_test_support import (
    MANUAL_PAGE_27,
    answer_reply,
    extract_reply,
    intent_reply,
    make_retriever,
)

import rag_agent.__main__ as cli
from rag_agent.providers.base import ModelAuthError
from rag_agent.providers.fake import FakeChatModel
from rag_agent.storage.business import BusinessRepository


@contextmanager
def dependencies(*replies: str) -> Iterator[tuple[object, FakeChatModel, BusinessRepository]]:
    with BusinessRepository(":memory:") as repository:
        repository.seed_from_file("data/business/seed.json")
        yield make_retriever(MANUAL_PAGE_27), FakeChatModel(list(replies)), repository


def install(monkeypatch: pytest.MonkeyPatch, *replies: str) -> None:
    monkeypatch.setattr(cli, "_open_agent", lambda settings: dependencies(*replies))


def test_agent_requires_a_question() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["agent"])

    assert excinfo.value.code == 2


def test_knowledge_turn_prints_the_trace(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch, intent_reply("knowledge"), answer_reply(MANUAL_PAGE_27))

    exit_code = cli.main(["agent", MANUAL_PAGE_27])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "answered"
    assert payload["intent"] == "knowledge"
    assert payload["answer"] == MANUAL_PAGE_27
    assert payload["trace"][0] == "classify:knowledge"


def test_device_turn_uses_the_simulated_store(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch, intent_reply("device"))

    exit_code = cli.main(
        ["agent", "D2002 还在保修吗", "--user-id", "U1001", "--device-id", "D2002"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "answered"
    assert payload["tool_results"][0]["tool"] == "device.lookup"


def test_ticket_turn_returns_a_pending_confirmation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(
        monkeypatch,
        intent_reply("ticket"),
        extract_reply(device_id="D2001", issue="主刷一直卡住", contact="138****0001"),
    )

    exit_code = cli.main(["agent", "帮我把 D2001 报修", "--user-id", "U1001"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "pending_confirmation"
    assert payload["pending_confirmation"]["confirmed"] is False
    assert payload["created_ticket"] is None


def test_confirm_flag_creates_the_ticket(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(
        monkeypatch,
        intent_reply("ticket"),
        extract_reply(device_id="D2001", issue="主刷一直卡住", contact="138****0001"),
    )

    exit_code = cli.main(["agent", "帮我把 D2001 报修", "--user-id", "U1001", "--confirm"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "ticket_created"
    assert payload["created_ticket"]["ticket_id"].startswith("T")


def test_low_confidence_intent_asks_for_clarification(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch, intent_reply("device", 0.1))

    exit_code = cli.main(["agent", "嗯"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "needs_input"
    assert payload["intent"] == "unknown"


def test_missing_credentials_are_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    @contextmanager
    def failing(settings: object) -> Iterator[tuple[object, ...]]:
        raise ModelAuthError("no API key configured")
        yield  # pragma: no cover

    monkeypatch.setattr(cli, "_open_agent", failing)

    exit_code = cli.main(["agent", "问题"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["code"] == "auth"
