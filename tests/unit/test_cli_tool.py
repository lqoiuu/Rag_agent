"""The tool command: contract listing, dispatch and exit codes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

import rag_agent.__main__ as cli
from rag_agent.storage.business import BusinessRepository


@pytest.fixture
def patched_repository(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Route the command at one in-memory repository for the whole test."""

    with BusinessRepository(":memory:") as repository:
        repository.seed_from_file("data/business/seed.json")

        @contextmanager
        def factory(path: object) -> Iterator[BusinessRepository]:
            yield repository

        monkeypatch.setattr(cli, "BusinessRepository", factory)
        yield


def test_tool_list_prints_the_contracts(
    patched_repository: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["tool", "list"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    names = [tool["name"] for tool in payload["tools"]]
    assert names == ["user.lookup", "device.lookup", "order.lookup", "ticket.create"]
    assert "模拟数据" in payload["note"]
    device_schema = next(
        tool["args_schema"] for tool in payload["tools"] if tool["name"] == "device.lookup"
    )
    assert "device_id" in device_schema["properties"]


def test_tool_list_is_the_default(
    patched_repository: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["tool"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(payload["tools"]) == 4


def test_user_lookup_returns_the_profile(
    patched_repository: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["tool", "user.lookup", "--args", '{"user_id": "U1001"}'])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["data"]["user"]["name"] == "张明"


def test_device_lookup_reports_warranty(
    patched_repository: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(
        [
            "tool",
            "device.lookup",
            "--args",
            '{"device_id": "D2002", "as_of": "2026-09-11"}',
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["data"]["device"]["warranty_active"] is True


def test_permission_failure_returns_exit_code_one(
    patched_repository: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(
        [
            "tool",
            "device.lookup",
            "--args",
            '{"device_id": "D2003", "user_id": "U1001"}',
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["error_code"] == "permission_denied"


def test_ticket_creation_without_confirmation_is_refused(
    patched_repository: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(
        [
            "tool",
            "ticket.create",
            "--args",
            '{"user_id": "U1001", "device_id": "D2001", '
            '"issue": "主刷一直卡住", "contact": "138****0001"}',
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["error_code"] == "confirmation_required"


def test_confirmed_ticket_creation_succeeds_and_is_idempotent(
    patched_repository: None, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = (
        '{"user_id": "U1001", "device_id": "D2001", "issue": "主刷一直卡住", '
        '"contact": "138****0001", "confirmed": true}'
    )

    first = cli.main(["tool", "ticket.create", "--args", arguments])
    first_payload = json.loads(capsys.readouterr().out)
    second = cli.main(["tool", "ticket.create", "--args", arguments])
    second_payload = json.loads(capsys.readouterr().out)

    assert first == 0 and second == 0
    assert first_payload["data"]["created"] is True
    assert second_payload["data"]["created"] is False
    assert (
        first_payload["data"]["ticket"]["ticket_id"]
        == second_payload["data"]["ticket"]["ticket_id"]
    )


def test_unknown_tool_is_rejected(
    patched_repository: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["tool", "device.delete"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["code"] == "unknown_tool"


def test_invalid_json_arguments_are_rejected(
    patched_repository: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["tool", "user.lookup", "--args", "{not json}"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["code"] == "invalid_argument"


def test_non_object_arguments_are_rejected(
    patched_repository: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["tool", "user.lookup", "--args", "[1, 2]"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["code"] == "invalid_argument"


def test_schema_violations_are_rejected_before_any_lookup(
    patched_repository: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["tool", "user.lookup", "--args", '{"user_id": ""}'])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["code"] == "invalid_argument"
