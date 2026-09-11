"""SQLite storage for the simulated business entities.

The data is deliberately predictable: it is seeded from a committed JSON file
instead of being generated randomly, so a demo, a test and a report all see the
same users, devices and orders. Nothing here represents a real person.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

BUSINESS_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        phone_masked TEXT NOT NULL,
        email TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS devices (
        device_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        model TEXT NOT NULL,
        serial TEXT NOT NULL,
        activated_on TEXT NOT NULL,
        warranty_months INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        device_id TEXT NOT NULL,
        purchased_on TEXT NOT NULL,
        amount_cents INTEGER NOT NULL,
        status TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tickets (
        ticket_id TEXT PRIMARY KEY,
        idempotency_key TEXT NOT NULL UNIQUE,
        user_id TEXT NOT NULL,
        device_id TEXT NOT NULL,
        issue TEXT NOT NULL,
        contact TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_devices_user ON devices (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_orders_user ON orders (user_id)",
)


@dataclass(frozen=True, slots=True)
class UserRecord:
    user_id: str
    name: str
    phone_masked: str
    email: str

    def as_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "phone_masked": self.phone_masked,
            "email": self.email,
        }


@dataclass(frozen=True, slots=True)
class DeviceRecord:
    device_id: str
    user_id: str
    model: str
    serial: str
    activated_on: date
    warranty_months: int

    @property
    def warranty_expires_on(self) -> date:
        return add_months(self.activated_on, self.warranty_months)

    def warranty_active(self, *, as_of: date) -> bool:
        return as_of <= self.warranty_expires_on

    def as_dict(self, *, as_of: date | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "device_id": self.device_id,
            "user_id": self.user_id,
            "model": self.model,
            "serial": self.serial,
            "activated_on": self.activated_on.isoformat(),
            "warranty_months": self.warranty_months,
            "warranty_expires_on": self.warranty_expires_on.isoformat(),
        }
        if as_of is not None:
            payload["warranty_active"] = self.warranty_active(as_of=as_of)
            payload["warranty_checked_on"] = as_of.isoformat()
        return payload


@dataclass(frozen=True, slots=True)
class OrderRecord:
    order_id: str
    user_id: str
    device_id: str
    purchased_on: date
    amount_cents: int
    status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "user_id": self.user_id,
            "device_id": self.device_id,
            "purchased_on": self.purchased_on.isoformat(),
            "amount_cents": self.amount_cents,
            "amount": f"{self.amount_cents / 100:.2f}",
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class TicketRecord:
    ticket_id: str
    idempotency_key: str
    user_id: str
    device_id: str
    issue: str
    contact: str
    status: str
    created_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "ticket_id": self.ticket_id,
            "user_id": self.user_id,
            "device_id": self.device_id,
            "issue": self.issue,
            "contact": self.contact,
            "status": self.status,
            "created_at": self.created_at,
        }


def add_months(value: date, months: int) -> date:
    """Add whole months, clamping the day to the target month's length."""

    total = value.month - 1 + months
    year = value.year + total // 12
    month = total % 12 + 1
    day = min(value.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - date(year, month, 1)).days


class BusinessRepository:
    """Typed access to the simulated business tables."""

    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        self._connection = sqlite3.connect(self._path)
        self._connection.row_factory = sqlite3.Row
        self.initialize()

    @property
    def path(self) -> str:
        return self._path

    def initialize(self) -> None:
        with self._connection:
            for statement in BUSINESS_SCHEMA:
                self._connection.execute(statement)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> BusinessRepository:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def seed_from_file(self, path: Path | str, *, replace: bool = False) -> int:
        """Load the seed file; repeated calls change nothing unless replacing."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        users = _as_list(payload.get("users"))
        devices = _as_list(payload.get("devices"))
        orders = _as_list(payload.get("orders"))
        verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"

        with self._connection:
            for user in users:
                self._connection.execute(
                    f"{verb} INTO users (user_id, name, phone_masked, email) VALUES (?, ?, ?, ?)",
                    (
                        str(user["user_id"]),
                        str(user["name"]),
                        str(user["phone_masked"]),
                        str(user["email"]),
                    ),
                )
            for device in devices:
                self._connection.execute(
                    f"""
                    {verb} INTO devices (
                        device_id, user_id, model, serial, activated_on, warranty_months
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(device["device_id"]),
                        str(device["user_id"]),
                        str(device["model"]),
                        str(device["serial"]),
                        str(device["activated_on"]),
                        int(device["warranty_months"]),
                    ),
                )
            for order in orders:
                self._connection.execute(
                    f"""
                    {verb} INTO orders (
                        order_id, user_id, device_id, purchased_on, amount_cents, status
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(order["order_id"]),
                        str(order["user_id"]),
                        str(order["device_id"]),
                        str(order["purchased_on"]),
                        int(order["amount_cents"]),
                        str(order["status"]),
                    ),
                )
        return len(users) + len(devices) + len(orders)

    def get_user(self, user_id: str) -> UserRecord | None:
        row = self._connection.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return None if row is None else _user_from_row(row)

    def get_device(self, device_id: str) -> DeviceRecord | None:
        row = self._connection.execute(
            "SELECT * FROM devices WHERE device_id = ?", (device_id,)
        ).fetchone()
        return None if row is None else _device_from_row(row)

    def list_devices_for_user(self, user_id: str) -> tuple[DeviceRecord, ...]:
        rows = self._connection.execute(
            "SELECT * FROM devices WHERE user_id = ? ORDER BY device_id", (user_id,)
        ).fetchall()
        return tuple(_device_from_row(row) for row in rows)

    def get_order(self, order_id: str) -> OrderRecord | None:
        row = self._connection.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        return None if row is None else _order_from_row(row)

    def list_orders_for_user(self, user_id: str) -> tuple[OrderRecord, ...]:
        rows = self._connection.execute(
            "SELECT * FROM orders WHERE user_id = ? ORDER BY order_id", (user_id,)
        ).fetchall()
        return tuple(_order_from_row(row) for row in rows)

    def find_ticket_by_key(self, idempotency_key: str) -> TicketRecord | None:
        row = self._connection.execute(
            "SELECT * FROM tickets WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        return None if row is None else _ticket_from_row(row)

    def get_ticket(self, ticket_id: str) -> TicketRecord | None:
        row = self._connection.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        return None if row is None else _ticket_from_row(row)

    def list_tickets(self) -> tuple[TicketRecord, ...]:
        rows = self._connection.execute("SELECT * FROM tickets ORDER BY ticket_id").fetchall()
        return tuple(_ticket_from_row(row) for row in rows)

    def create_ticket(
        self,
        *,
        ticket_id: str,
        idempotency_key: str,
        user_id: str,
        device_id: str,
        issue: str,
        contact: str,
        created_at: str,
    ) -> TicketRecord:
        """Insert one ticket; the unique idempotency key prevents duplicates."""

        with self._connection:
            self._connection.execute(
                """
                INSERT INTO tickets (
                    ticket_id, idempotency_key, user_id, device_id, issue, contact,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    idempotency_key,
                    user_id,
                    device_id,
                    issue,
                    contact,
                    "open",
                    created_at,
                ),
            )
        return TicketRecord(
            ticket_id=ticket_id,
            idempotency_key=idempotency_key,
            user_id=user_id,
            device_id=device_id,
            issue=issue,
            contact=contact,
            status="open",
            created_at=created_at,
        )


def _as_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _user_from_row(row: sqlite3.Row) -> UserRecord:
    return UserRecord(
        user_id=str(row["user_id"]),
        name=str(row["name"]),
        phone_masked=str(row["phone_masked"]),
        email=str(row["email"]),
    )


def _device_from_row(row: sqlite3.Row) -> DeviceRecord:
    return DeviceRecord(
        device_id=str(row["device_id"]),
        user_id=str(row["user_id"]),
        model=str(row["model"]),
        serial=str(row["serial"]),
        activated_on=date.fromisoformat(str(row["activated_on"])),
        warranty_months=int(row["warranty_months"]),
    )


def _order_from_row(row: sqlite3.Row) -> OrderRecord:
    return OrderRecord(
        order_id=str(row["order_id"]),
        user_id=str(row["user_id"]),
        device_id=str(row["device_id"]),
        purchased_on=date.fromisoformat(str(row["purchased_on"])),
        amount_cents=int(row["amount_cents"]),
        status=str(row["status"]),
    )


def _ticket_from_row(row: sqlite3.Row) -> TicketRecord:
    return TicketRecord(
        ticket_id=str(row["ticket_id"]),
        idempotency_key=str(row["idempotency_key"]),
        user_id=str(row["user_id"]),
        device_id=str(row["device_id"]),
        issue=str(row["issue"]),
        contact=str(row["contact"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
    )
