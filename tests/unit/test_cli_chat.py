"""The chat and thread commands: persisted threads, isolation and confirmations."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from agent_test_support import (
    MANUAL_PAGE_27,
    answer_reply,
    extract_reply,
    intent_reply,
    make_retriever,
)

import rag_agent.__main__ as cli
from rag_agent.config.settings import Settings
from rag_agent.providers.base import ModelAuthError
from rag_agent.providers.fake import FakeChatModel


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[Settings]:
    """Settings pointing at an empty index and a per-test conversation database."""

    settings = Settings(
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "chroma",
        sqlite_path=tmp_path / "rag_agent.sqlite3",
        checkpoint_path=tmp_path / "rag_agent.sqlite3",
        qwen_api_key="test-key",
    )
    (tmp_path / "data" / "business").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "business" / "seed.json").write_text(
        Path("data/business/seed.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    yield settings


class _ManagedChatModel(FakeChatModel):
    """The real wiring opens the chat model as a context manager; the fake is not one."""

    def __enter__(self) -> _ManagedChatModel:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def install(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    *replies: str,
    model: FakeChatModel | None = None,
) -> None:
    """Use the real command wiring, with the chat model replaced by a script.

    ``_open_conversation`` is deliberately *not* stubbed: the test wants the real
    checkerpointer, the real thread registry and the real graph, so that "the thread
    survived" is actually being demonstrated. Only the model — the one thing that
    would need the network — is scripted.
    """

    scripted = model or _ManagedChatModel(list(replies))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "build_chat_model", lambda settings=None, **_: scripted)
    monkeypatch.setattr(cli, "Retriever", lambda **_: make_retriever(MANUAL_PAGE_27))


def test_chat_requires_a_question_or_a_decision() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["chat", "--thread-id", "T1", "--user-id", "U1001"])

    assert excinfo.value.code == 2


def test_confirm_and_cancel_cannot_be_combined() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["chat", "问题", "--confirm", "--cancel"])

    assert excinfo.value.code == 2


def test_window_must_be_positive() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["chat", "问题", "--window", "0"])

    assert excinfo.value.code == 2


def test_chat_answers_and_generates_a_thread_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    install(monkeypatch, workspace, intent_reply("knowledge"), answer_reply(MANUAL_PAGE_27))

    exit_code = cli.main(["chat", MANUAL_PAGE_27, "--user-id", "U1001"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "answered"
    assert payload["thread_id"].startswith("T")
    assert payload["memory"]["checkpoints_after"] > 0


def test_thread_list_shows_the_generated_thread(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    install(monkeypatch, workspace, intent_reply("knowledge"), answer_reply(MANUAL_PAGE_27))
    cli.main(["chat", MANUAL_PAGE_27, "--user-id", "U1001", "--thread-id", "T1"])
    capsys.readouterr()

    exit_code = cli.main(["thread", "list", "--user-id", "U1001"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["thread_count"] == 1
    assert payload["threads"][0]["thread_id"] == "T1"
    assert payload["threads"][0]["checkpoint_count"] > 0


def test_a_second_invocation_continues_the_thread(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    """Two separate ``cli.main`` calls share one conversation through SQLite."""

    scripted = _ManagedChatModel(
        [
            intent_reply("knowledge"),
            answer_reply(MANUAL_PAGE_27),
            intent_reply("knowledge"),
            answer_reply(MANUAL_PAGE_27),
        ]
    )
    install(monkeypatch, workspace, model=scripted)
    cli.main(["chat", "第一轮问题", "--user-id", "U1001", "--thread-id", "T1"])
    capsys.readouterr()

    exit_code = cli.main(["chat", "第二轮问题", "--user-id", "U1001", "--thread-id", "T1"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "answered"
    assert payload["memory"]["window_size"] == 2  # 上一轮的用户消息与助手回复


def test_another_user_is_refused_on_someone_elses_thread(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    install(monkeypatch, workspace, intent_reply("knowledge"), answer_reply(MANUAL_PAGE_27))
    cli.main(["chat", "问题", "--user-id", "U1001", "--thread-id", "T1"])
    capsys.readouterr()

    exit_code = cli.main(["chat", "偷看", "--user-id", "U1002", "--thread-id", "T1"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["code"] == "thread_ownership_conflict"


def test_pending_ticket_is_not_written_until_confirmed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    install(
        monkeypatch,
        workspace,
        intent_reply("ticket"),
        extract_reply(device_id="D2001", issue="主刷一直卡住", contact="138****0001"),
    )

    paused = cli.main(
        [
            "chat",
            "帮我把 D2001 报修，主刷一直卡住，联系我 138****0001",
            "--user-id",
            "U1001",
            "--thread-id",
            "T1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert paused == 1
    assert payload["status"] == "pending_confirmation"
    assert payload["paused"] is True
    assert payload["pending_action"]["action"] == "ticket.create"
    assert payload["created_ticket"] is None

    from rag_agent.storage.business import BusinessRepository

    with BusinessRepository(workspace.sqlite_path) as repository:
        assert repository.list_tickets() == ()


def test_cancel_after_a_pause_does_not_write(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    install(
        monkeypatch,
        workspace,
        intent_reply("ticket"),
        extract_reply(device_id="D2001", issue="主刷一直卡住", contact="138****0001"),
    )
    cli.main(
        [
            "chat",
            "帮我把 D2001 报修，主刷一直卡住，联系我 138****0001",
            "--user-id",
            "U1001",
            "--thread-id",
            "T1",
        ]
    )
    capsys.readouterr()

    exit_code = cli.main(["chat", "--cancel", "--user-id", "U1001", "--thread-id", "T1"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "ticket_cancelled"


def test_confirm_after_a_pause_creates_the_ticket(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    install(
        monkeypatch,
        workspace,
        intent_reply("ticket"),
        extract_reply(device_id="D2001", issue="主刷一直卡住", contact="138****0001"),
    )
    cli.main(
        [
            "chat",
            "帮我把 D2001 报修，主刷一直卡住，联系我 138****0001",
            "--user-id",
            "U1001",
            "--thread-id",
            "T1",
        ]
    )
    capsys.readouterr()

    exit_code = cli.main(["chat", "--confirm", "--user-id", "U1001", "--thread-id", "T1"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "ticket_created"
    assert payload["created_ticket"]["ticket_id"].startswith("T")


def test_thread_clear_removes_the_conversation_but_keeps_preferences(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    install(monkeypatch, workspace, intent_reply("knowledge"), answer_reply(MANUAL_PAGE_27))
    cli.main(
        [
            "chat",
            MANUAL_PAGE_27,
            "--user-id",
            "U1001",
            "--thread-id",
            "T1",
            "--preference",
            "contact=email",
        ]
    )
    capsys.readouterr()

    exit_code = cli.main(["thread", "clear", "--thread-id", "T1", "--user-id", "U1001"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["checkpoints_removed"] > 0

    cli.main(["thread", "list", "--user-id", "U1001"])
    listed = json.loads(capsys.readouterr().out)
    assert listed["thread_count"] == 0

    cli.main(["thread", "preferences", "--user-id", "U1001"])
    prefs = json.loads(capsys.readouterr().out)
    assert prefs["preferences"] == {"contact": "email"}
    assert "contact：email" in prefs["rendered"]


def test_clearing_someone_elses_thread_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    install(monkeypatch, workspace, intent_reply("knowledge"), answer_reply(MANUAL_PAGE_27))
    cli.main(["chat", "问题", "--user-id", "U1001", "--thread-id", "T1"])
    capsys.readouterr()

    exit_code = cli.main(["thread", "clear", "--thread-id", "T1", "--user-id", "U1002"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["code"] == "thread_ownership_conflict"


def test_clear_without_a_thread_id_is_an_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    install(monkeypatch, workspace)

    exit_code = cli.main(["thread", "clear", "--user-id", "U1001"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["code"] == "invalid_argument"


def test_unknown_thread_action_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    install(monkeypatch, workspace)

    exit_code = cli.main(["thread", "frobnicate"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["code"] == "unknown_action"


def test_chat_reports_turn_metrics(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    """A turn must report what it measured, not only what it answered."""

    install(monkeypatch, workspace, intent_reply("knowledge"), answer_reply(MANUAL_PAGE_27))

    cli.main(["chat", MANUAL_PAGE_27, "--user-id", "U1001", "--thread-id", "T1"])
    payload = json.loads(capsys.readouterr().out)

    metrics = payload["metrics"]
    assert metrics is not None
    assert metrics["total_ms"] > 0
    assert metrics["retrieval"]["hits"] > 0
    assert metrics["tools"] == {
        "calls": 0,
        "failures": 0,
        "success_rate": 1.0,
        "error_codes": [],
    }
    assert metrics["injection_suspected"] == []


def test_chat_exposes_the_turn_separately_from_the_accumulated_thread(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    """The payload must let a reader tell this turn's work from the thread's history."""

    install(monkeypatch, workspace, intent_reply("device"), intent_reply("device"))
    cli.main(["chat", "D2002 还在保修吗", "--user-id", "U1001", "--thread-id", "T1"])
    capsys.readouterr()

    exit_code = cli.main(["chat", "D2001 还在保修吗", "--user-id", "U1001", "--thread-id", "T1"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["turn"]["trace"] == ["classify:device", "device:ok"]
    assert len(payload["turn"]["tool_results"]) == 2
    assert len(payload["trace"]) > len(payload["turn"]["trace"])
    assert len(payload["tool_results"]) > len(payload["turn"]["tool_results"])


def test_a_support_agent_reaches_the_device_tool_for_a_foreign_device(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    """``--role`` is the caller's declaration, and the role has to actually do something.

    The point asserted here is the *permission* outcome: with ``end_user`` the tool refuses
    before reading anything, so the error code is ``permission_denied``. The other assertions
    are deliberately weaker — the caller id does not exist in the simulated data, so the
    lookup may legitimately report ``not_found``, and pretending otherwise would be asserting
    the fixture rather than the behaviour.
    """

    install(monkeypatch, workspace, intent_reply("device"), intent_reply("device"))

    cli.main(
        [
            "chat",
            "D2003 还在保修吗",
            "--user-id",
            "U1002",
            "--thread-id",
            "T-support",
            "--role",
            "support_agent",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["error_code"] != "permission_denied"
    assert (payload["tool_results"] or [{}])[0]["tool"] == "device.lookup"


def test_a_plain_user_is_denied_another_users_device(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    """An undeclared role stays the strict default, so a foreign device is refused.

    The code is deliberately not pinned: ``not_found`` and ``permission_denied`` are both
    legitimate refusals here, and asserting one would be asserting the fixture rather than the
    permission behaviour.
    """

    install(monkeypatch, workspace, intent_reply("device"), intent_reply("device"))

    exit_code = cli.main(
        ["chat", "D2003 还在保修吗", "--user-id", "U9999", "--thread-id", "T-denied"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["error_code"] in {"permission_denied", "not_found"}
    assert payload["status"] == "error"


def test_an_unknown_role_falls_back_to_the_least_privileged_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    """A typo in ``--role`` must not be read as a promotion."""

    install(monkeypatch, workspace, intent_reply("device"))

    exit_code = cli.main(
        [
            "chat",
            "D2003 还在保修吗",
            "--user-id",
            "U9999",
            "--thread-id",
            "T-typo",
            "--role",
            "administrator",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["error_code"] == "permission_denied"


def test_bad_preference_syntax_is_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    install(monkeypatch, workspace, intent_reply("knowledge"), answer_reply(MANUAL_PAGE_27))

    exit_code = cli.main(["chat", "问题", "--user-id", "U1001", "--preference", "contact"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["code"] == "invalid_argument"


def test_missing_credentials_are_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Settings
) -> None:
    def failing(*_: object, **__: object) -> None:
        raise ModelAuthError("no API key configured")

    monkeypatch.setattr(cli, "get_settings", lambda: workspace)
    monkeypatch.setattr(cli, "build_chat_model", failing)

    exit_code = cli.main(["chat", "问题", "--user-id", "U1001", "--thread-id", "T1"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["code"] == "auth"


def test_agent_command_no_longer_accepts_confirm(
    monkeypatch: pytest.MonkeyPatch, workspace: Settings
) -> None:
    """Resuming a paused run belongs to ``chat``; ``agent`` stays stateless."""

    install(monkeypatch, workspace, intent_reply("ticket"))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["agent", "报修", "--confirm"])

    assert excinfo.value.code == 2
