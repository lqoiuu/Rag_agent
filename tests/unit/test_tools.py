"""Tool contracts: schema validation, permissions, idempotency and failures."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import pytest

from rag_agent.storage.business import BusinessRepository
from rag_agent.tools import (
    CreateTicketArgs,
    DeviceLookupArgs,
    OrderLookupArgs,
    UserLookupArgs,
    create_ticket,
    device_lookup,
    order_lookup,
    user_lookup,
)

AS_OF = date(2026, 9, 11)


@pytest.fixture
def repository() -> Iterator[BusinessRepository]:
    with BusinessRepository(":memory:") as opened:
        opened.seed_from_file("data/business/seed.json")
        yield opened


def test_user_lookup_returns_the_masked_profile(repository: BusinessRepository) -> None:
    result = user_lookup(UserLookupArgs(user_id="U1001"), repository=repository)

    assert result.ok is True
    assert result.tool == "user.lookup"
    assert result.request_id.startswith("req-")
    assert result.data is not None
    assert result.data["user"] == {
        "user_id": "U1001",
        "name": "张明",
        "phone_masked": "138****0001",
        "email": "user1001@example.com",
    }


def test_unknown_user_is_a_classified_failure(repository: BusinessRepository) -> None:
    result = user_lookup(UserLookupArgs(user_id="U9999"), repository=repository)

    assert result.status == "error"
    assert result.error_code == "not_found"
    assert result.retryable is False
    assert result.data is None
    assert "U9999" in (result.error_message or "")


def test_device_lookup_reports_warranty_for_a_fixed_date(
    repository: BusinessRepository,
) -> None:
    result = device_lookup(DeviceLookupArgs(device_id="D2001", as_of=AS_OF), repository=repository)

    assert result.data is not None
    device = result.data["device"]
    assert device["warranty_expires_on"] == "2026-03-15"  # type: ignore[index]
    assert device["warranty_active"] is False  # type: ignore[index]
    assert result.data["as_of"] == "2026-09-11"


def test_device_lookup_can_list_a_users_devices(repository: BusinessRepository) -> None:
    result = device_lookup(DeviceLookupArgs(user_id="U1002", as_of=AS_OF), repository=repository)

    assert result.data is not None
    assert result.data["count"] == 2
    assert [item["device_id"] for item in result.data["devices"]] == ["D2003", "D2005"]  # type: ignore[index]


def test_device_lookup_requires_a_selector() -> None:
    with pytest.raises(ValueError, match="device_id or user_id"):
        DeviceLookupArgs()


def test_device_of_another_user_is_denied(repository: BusinessRepository) -> None:
    result = device_lookup(
        DeviceLookupArgs(device_id="D2003", user_id="U1001", as_of=AS_OF), repository=repository
    )

    assert result.status == "error"
    assert result.error_code == "permission_denied"


def test_order_lookup_by_id_and_by_user(repository: BusinessRepository) -> None:
    single = order_lookup(OrderLookupArgs(order_id="O3001"), repository=repository)
    listed = order_lookup(OrderLookupArgs(user_id="U1001"), repository=repository)

    assert single.data is not None
    assert single.data["order"]["order_id"] == "O3001"  # type: ignore[index]
    assert listed.data is not None
    assert listed.data["count"] == 2


def test_order_of_another_user_is_denied(repository: BusinessRepository) -> None:
    result = order_lookup(OrderLookupArgs(order_id="O3003", user_id="U1001"), repository=repository)

    assert result.error_code == "permission_denied"


def test_ticket_creation_requires_confirmation(repository: BusinessRepository) -> None:
    result = create_ticket(
        CreateTicketArgs(
            user_id="U1001",
            device_id="D2001",
            issue="主刷一直卡住，无法正常清扫",
            contact="138****0001",
        ),
        repository=repository,
    )

    assert result.status == "error"
    assert result.error_code == "confirmation_required"
    assert repository.list_tickets() == ()


def test_confirmed_ticket_creation_succeeds(repository: BusinessRepository) -> None:
    result = create_ticket(
        CreateTicketArgs(
            user_id="U1001",
            device_id="D2001",
            issue="主刷一直卡住，无法正常清扫",
            contact="138****0001",
            confirmed=True,
        ),
        repository=repository,
    )

    assert result.ok is True
    assert result.data is not None
    assert result.data["created"] is True
    ticket = result.data["ticket"]
    assert ticket["status"] == "open"  # type: ignore[index]
    assert ticket["ticket_id"].startswith("T")  # type: ignore[index]
    assert repository.list_tickets()[0].ticket_id == ticket["ticket_id"]  # type: ignore[index]


def test_repeating_the_same_request_returns_the_same_ticket(
    repository: BusinessRepository,
) -> None:
    args = CreateTicketArgs(
        user_id="U1001",
        device_id="D2001",
        issue="主刷一直卡住",
        contact="138****0001",
        confirmed=True,
    )

    first = create_ticket(args, repository=repository)
    second = create_ticket(args, repository=repository)

    assert first.data is not None and second.data is not None
    assert first.data["ticket"]["ticket_id"] == second.data["ticket"]["ticket_id"]  # type: ignore[index]
    assert first.data["created"] is True
    assert second.data["created"] is False
    assert len(repository.list_tickets()) == 1


def test_a_caller_supplied_idempotency_key_is_honoured(
    repository: BusinessRepository,
) -> None:
    first = create_ticket(
        CreateTicketArgs(
            user_id="U1001",
            device_id="D2001",
            issue="第一次描述",
            contact="138****0001",
            confirmed=True,
            idempotency_key="order-42-attempt-1",
        ),
        repository=repository,
    )
    second = create_ticket(
        CreateTicketArgs(
            user_id="U1001",
            device_id="D2001",
            issue="描述被改写了一点",
            contact="138****0001",
            confirmed=True,
            idempotency_key="order-42-attempt-1",
        ),
        repository=repository,
    )

    assert first.data is not None and second.data is not None
    assert first.data["ticket"]["ticket_id"] == second.data["ticket"]["ticket_id"]  # type: ignore[index]
    assert len(repository.list_tickets()) == 1


def test_ticket_for_another_users_device_is_denied(repository: BusinessRepository) -> None:
    result = create_ticket(
        CreateTicketArgs(
            user_id="U1001",
            device_id="D2003",
            issue="主刷一直卡住",
            contact="138****0001",
            confirmed=True,
        ),
        repository=repository,
    )

    assert result.error_code == "permission_denied"
    assert repository.list_tickets() == ()


def test_ticket_for_unknown_device_is_not_found(repository: BusinessRepository) -> None:
    result = create_ticket(
        CreateTicketArgs(
            user_id="U1001",
            device_id="D9999",
            issue="主刷一直卡住",
            contact="138****0001",
            confirmed=True,
        ),
        repository=repository,
    )

    assert result.error_code == "not_found"


def test_too_short_issue_is_rejected_by_the_schema() -> None:
    with pytest.raises(ValueError):
        CreateTicketArgs(user_id="U1001", device_id="D2001", issue="卡住", contact="138****0001")


def test_invalid_arguments_are_reported_as_a_tool_result(
    repository: BusinessRepository,
) -> None:
    args = CreateTicketArgs.model_construct(
        user_id="U1001",
        device_id="D2001",
        issue="卡住",  # 违反最小长度，但绕过校验构造出来
        contact="138****0001",
        confirmed=True,
        idempotency_key=None,
    )

    result = create_ticket(args, repository=repository)

    assert result.status == "error"
    assert result.error_code == "invalid_argument"


def test_store_failure_is_retryable_and_not_a_normal_result(
    repository: BusinessRepository,
) -> None:
    repository.close()

    result = user_lookup(UserLookupArgs(user_id="U1001"), repository=repository)

    assert result.status == "error"
    assert result.error_code == "unavailable"
    assert result.retryable is True
    assert result.data is None


def test_results_never_carry_free_form_errors_as_data(
    repository: BusinessRepository,
) -> None:
    result = user_lookup(UserLookupArgs(user_id="U9999"), repository=repository)

    assert result.data is None
    assert result.error_code is not None
    assert result.model_dump()["status"] == "error"
