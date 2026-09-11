"""Contract-bound business tools over the simulated repository.

Every tool takes validated arguments and a repository, and returns a
:class:`ToolResult`: failures become ``status="error"`` with a stable code rather
than an exception or a sentence a model could mistake for an answer.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import UTC, date, datetime

from pydantic import ValidationError

from rag_agent.storage.business import BusinessRepository, DeviceRecord, UserRecord
from rag_agent.tools.errors import (
    ConfirmationRequiredError,
    InvalidArgumentError,
    NotFoundError,
    PermissionDeniedError,
    ToolError,
    ToolUnavailableError,
)
from rag_agent.tools.models import (
    CreateTicketArgs,
    DeviceLookupArgs,
    OrderLookupArgs,
    ToolResult,
    UserLookupArgs,
)

LOGGER = logging.getLogger(__name__)

USER_LOOKUP = "user.lookup"
DEVICE_LOOKUP = "device.lookup"
ORDER_LOOKUP = "order.lookup"
TICKET_CREATE = "ticket.create"

TOOL_NAMES = (USER_LOOKUP, DEVICE_LOOKUP, ORDER_LOOKUP, TICKET_CREATE)


def new_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:12]}"


def user_lookup(args: UserLookupArgs, *, repository: BusinessRepository) -> ToolResult:
    """Return one simulated user, or a classified failure."""

    return _run(USER_LOOKUP, lambda: {"user": _require_user(repository, args.user_id).as_dict()})


def device_lookup(args: DeviceLookupArgs, *, repository: BusinessRepository) -> ToolResult:
    """Return one device or all devices of a user, with warranty status."""

    def action() -> dict[str, object]:
        as_of = args.as_of or _today()
        if args.device_id is not None:
            device = _require_device(repository, args.device_id)
            if args.user_id is not None:
                _require_owner(device, args.user_id)
            return {"device": device.as_dict(as_of=as_of), "as_of": as_of.isoformat()}

        assert args.user_id is not None  # 由 DeviceLookupArgs 保证
        _require_user(repository, args.user_id)
        devices = repository.list_devices_for_user(args.user_id)
        return {
            "user_id": args.user_id,
            "devices": [device.as_dict(as_of=as_of) for device in devices],
            "count": len(devices),
            "as_of": as_of.isoformat(),
        }

    return _run(DEVICE_LOOKUP, action)


def order_lookup(args: OrderLookupArgs, *, repository: BusinessRepository) -> ToolResult:
    """Return one order or all orders of a user."""

    def action() -> dict[str, object]:
        if args.order_id is not None:
            order = repository.get_order(args.order_id)
            if order is None:
                raise NotFoundError(f"order {args.order_id!r} does not exist")
            if args.user_id is not None and order.user_id != args.user_id:
                raise PermissionDeniedError(
                    f"order {order.order_id!r} does not belong to user {args.user_id!r}"
                )
            return {"order": order.as_dict()}

        assert args.user_id is not None  # 由 OrderLookupArgs 保证
        _require_user(repository, args.user_id)
        orders = repository.list_orders_for_user(args.user_id)
        return {
            "user_id": args.user_id,
            "orders": [order.as_dict() for order in orders],
            "count": len(orders),
        }

    return _run(ORDER_LOOKUP, action)


def create_ticket(args: CreateTicketArgs, *, repository: BusinessRepository) -> ToolResult:
    """Create one simulated ticket, but only after explicit confirmation.

    Repeating the same request returns the existing ticket instead of creating a
    second one, which is what makes a retry safe.
    """

    def action() -> dict[str, object]:
        draft = args.draft()
        if not args.confirmed:
            raise ConfirmationRequiredError(
                "ticket creation needs explicit user confirmation of the draft"
            )
        _require_user(repository, draft.user_id)
        device = _require_device(repository, draft.device_id)
        _require_owner(device, draft.user_id)

        key = args.idempotency_key or draft.idempotency_key()
        existing = repository.find_ticket_by_key(key)
        if existing is not None:
            LOGGER.info("ticket create deduplicated key=%s ticket=%s", key, existing.ticket_id)
            return {"ticket": existing.as_dict(), "created": False, "idempotency_key": key}

        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        record = repository.create_ticket(
            ticket_id=_new_ticket_id(),
            idempotency_key=key,
            user_id=draft.user_id,
            device_id=draft.device_id,
            issue=" ".join(draft.issue.split()),
            contact=" ".join(draft.contact.split()),
            created_at=created_at,
        )
        LOGGER.info("ticket created key=%s ticket=%s", key, record.ticket_id)
        return {"ticket": record.as_dict(), "created": True, "idempotency_key": key}

    return _run(TICKET_CREATE, action)


def _run(tool: str, action: object) -> ToolResult:
    """Execute one tool action and translate failures into a result."""

    request_id = new_request_id()
    try:
        data = action() if callable(action) else {}
    except ValidationError as exc:
        return _failure(tool, request_id, InvalidArgumentError(f"invalid arguments: {exc}"))
    except ToolError as exc:
        return _failure(tool, request_id, exc)
    except sqlite3.Error as exc:
        LOGGER.warning("tool %s store failure: %s", tool, exc)
        return _failure(tool, request_id, ToolUnavailableError(f"business store failed: {exc}"))
    return ToolResult(tool=tool, status="ok", request_id=request_id, data=dict(data))


def _failure(tool: str, request_id: str, error: ToolError) -> ToolResult:
    LOGGER.info("tool %s failed code=%s", tool, error.code)
    return ToolResult(
        tool=tool,
        status="error",
        request_id=request_id,
        error_code=error.code,
        error_message=error.message,
        retryable=error.retryable,
    )


def _require_user(repository: BusinessRepository, user_id: str) -> UserRecord:
    user = repository.get_user(user_id)
    if user is None:
        raise NotFoundError(f"user {user_id!r} does not exist")
    return user


def _require_device(repository: BusinessRepository, device_id: str) -> DeviceRecord:
    device = repository.get_device(device_id)
    if device is None:
        raise NotFoundError(f"device {device_id!r} does not exist")
    return device


def _require_owner(device: DeviceRecord, user_id: str) -> None:
    if device.user_id != user_id:
        raise PermissionDeniedError(
            f"device {device.device_id!r} does not belong to user {user_id!r}"
        )


def _today() -> date:
    return datetime.now(UTC).date()


def _new_ticket_id() -> str:
    return f"T{uuid.uuid4().hex[:8].upper()}"
