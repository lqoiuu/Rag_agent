"""Simulated business repository: predictable data, typed lookups."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import pytest

from rag_agent.storage.business import BusinessRepository, add_months


@pytest.fixture
def repository() -> Iterator[BusinessRepository]:
    with BusinessRepository(":memory:") as opened:
        opened.seed_from_file("data/business/seed.json")
        yield opened


def test_seed_loading_is_idempotent(repository: BusinessRepository) -> None:
    first = repository.seed_from_file("data/business/seed.json")
    second = repository.seed_from_file("data/business/seed.json")

    assert first == second == 12
    assert len(repository.list_devices_for_user("U1001")) == 2


def test_seed_can_replace_existing_rows(repository: BusinessRepository) -> None:
    repository.seed_from_file("data/business/seed.json", replace=True)

    user = repository.get_user("U1001")
    assert user is not None
    assert user.name == "张明"
    assert user.phone_masked == "138****0001"


def test_lookups_return_typed_records(repository: BusinessRepository) -> None:
    device = repository.get_device("D2001")
    order = repository.get_order("O3001")

    assert device is not None
    assert device.model == "DBX23"
    assert device.activated_on == date(2024, 3, 15)
    assert device.warranty_expires_on == date(2026, 3, 15)
    assert order is not None
    assert order.amount_cents == 289900
    assert order.as_dict()["amount"] == "2899.00"


def test_missing_entities_return_none(repository: BusinessRepository) -> None:
    assert repository.get_user("U9999") is None
    assert repository.get_device("D9999") is None
    assert repository.get_order("O9999") is None
    assert repository.get_ticket("T9999") is None


def test_warranty_status_depends_on_the_reference_date(
    repository: BusinessRepository,
) -> None:
    expired = repository.get_device("D2001")
    active = repository.get_device("D2002")

    assert expired is not None and active is not None
    assert expired.warranty_active(as_of=date(2026, 9, 11)) is False
    assert expired.warranty_active(as_of=date(2025, 1, 1)) is True
    assert active.warranty_active(as_of=date(2026, 9, 11)) is True


def test_devices_are_listed_in_a_stable_order(repository: BusinessRepository) -> None:
    devices = repository.list_devices_for_user("U1001")

    assert [device.device_id for device in devices] == ["D2001", "D2002"]


def test_orders_are_listed_for_a_user(repository: BusinessRepository) -> None:
    orders = repository.list_orders_for_user("U1001")

    assert [order.order_id for order in orders] == ["O3001", "O3002"]


def test_ticket_creation_and_idempotency_lookup(repository: BusinessRepository) -> None:
    ticket = repository.create_ticket(
        ticket_id="TABC12345",
        idempotency_key="idem-1",
        user_id="U1001",
        device_id="D2001",
        issue="主刷卡住",
        contact="138****0001",
        created_at="2026-09-11T00:00:00+00:00",
    )

    assert ticket.status == "open"
    assert repository.get_ticket("TABC12345") == ticket
    assert repository.find_ticket_by_key("idem-1") == ticket
    assert repository.list_tickets() == (ticket,)


def test_duplicate_idempotency_key_is_rejected(repository: BusinessRepository) -> None:
    import sqlite3

    repository.create_ticket(
        ticket_id="T1",
        idempotency_key="same",
        user_id="U1001",
        device_id="D2001",
        issue="主刷卡住",
        contact="138****0001",
        created_at="2026-09-11T00:00:00+00:00",
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.create_ticket(
            ticket_id="T2",
            idempotency_key="same",
            user_id="U1001",
            device_id="D2001",
            issue="另一件事",
            contact="138****0001",
            created_at="2026-09-11T00:00:00+00:00",
        )


def test_add_months_clamps_the_day() -> None:
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2023, 1, 31), 1) == date(2023, 2, 28)
    assert add_months(date(2024, 11, 30), 3) == date(2025, 2, 28)
    assert add_months(date(2024, 3, 15), 24) == date(2026, 3, 15)
    assert add_months(date(2024, 12, 15), 1) == date(2025, 1, 15)
