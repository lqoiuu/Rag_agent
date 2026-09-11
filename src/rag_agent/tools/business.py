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
from rag_agent.tools.permissions import ToolPermissions

LOGGER = logging.getLogger(__name__)

USER_LOOKUP = "user.lookup"
DEVICE_LOOKUP = "device.lookup"
ORDER_LOOKUP = "order.lookup"
TICKET_CREATE = "ticket.create"

TOOL_NAMES = (USER_LOOKUP, DEVICE_LOOKUP, ORDER_LOOKUP, TICKET_CREATE)


def new_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:12]}"


def user_lookup(
    args: UserLookupArgs,
    *,
    repository: BusinessRepository,
    permissions: ToolPermissions | None = None,
) -> ToolResult:
    """Return one simulated user, or a classified failure."""

    return _run(
        USER_LOOKUP,
        permissions=permissions,
        action=lambda: {"user": _require_user(repository, args.user_id).as_dict()},
    )


def device_lookup(
    args: DeviceLookupArgs,
    *,
    repository: BusinessRepository,
    permissions: ToolPermissions | None = None,
) -> ToolResult:
    """Return one device or all devices of a user, with warranty status."""

    def action() -> dict[str, object]:
        as_of = args.as_of or _today()
        if args.device_id is not None:
            device = _require_device(repository, args.device_id)
            if args.user_id is not None:
                _require_owner(device, args.user_id, permissions)
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

    return _run(DEVICE_LOOKUP, action, permissions=permissions)


def order_lookup(
    args: OrderLookupArgs,
    *,
    repository: BusinessRepository,
    permissions: ToolPermissions | None = None,
) -> ToolResult:
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

    return _run(ORDER_LOOKUP, action, permissions=permissions)


def create_ticket(
    args: CreateTicketArgs,
    *,
    repository: BusinessRepository,
    permissions: ToolPermissions | None = None,
) -> ToolResult:
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
        _require_owner(device, draft.user_id, permissions)

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

    return _run(TICKET_CREATE, action, permissions=permissions)


def _run(
    tool: str,
    action: object,
    *,
    permissions: ToolPermissions | None = None,
) -> ToolResult:
    """Execute one tool action and translate failures into a result.

    The permission check is the first thing that happens, before the action closure can
    touch a repository: a denied call must not be able to cause a read or a write that a
    later check would have to undo. It also runs through the same result path as every other
    failure, so a denial is a classified outcome rather than an exception the caller has to
    remember to catch.
    """

    request_id = new_request_id()
    role = str(permissions.role) if permissions is not None else "unspecified"
    try:
        if permissions is not None:
            permissions.require_tool(tool)
        data = action() if callable(action) else {}
    except ValidationError as exc:
        return _failure(tool, request_id, InvalidArgumentError(f"invalid arguments: {exc}"))
    except ToolError as exc:
        if isinstance(exc, PermissionDeniedError):
            LOGGER.warning("tool %s denied role=%s reason=%s", tool, role, exc.message)
        return _failure(tool, request_id, exc)
    except sqlite3.Error as exc:
        LOGGER.warning("tool %s store failure: %s", tool, exc)
        return _failure(tool, request_id, ToolUnavailableError(f"business store failed: {exc}"))
    LOGGER.info("tool %s ok role=%s", tool, role)
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


def _require_owner(
    device: DeviceRecord,
    user_id: str,
    permissions: ToolPermissions | None = None,
) -> None:
    """Refuse access to somebody else's device unless the caller's role allows it.

    The role check comes first and the owner check second, so the two failures stay
    distinguishable: "this role may never read across users" is a different event from "this
    argument claims a different owner". The default when no permissions are supplied is the
    strict behaviour, so a caller that forgets to pass them does not accidentally gain the
    cross-user read.
    """

    if permissions is not None:
        permissions.require_own_record(owner_user_id=device.user_id, tool_name=DEVICE_LOOKUP)
        if permissions.may_read_other_users:
            return
    if device.user_id != user_id:
        raise PermissionDeniedError(
            f"device {device.device_id!r} does not belong to user {user_id!r}"
        )


def _today() -> date:
    return datetime.now(UTC).date()


def _new_ticket_id() -> str:
    return f"T{uuid.uuid4().hex[:8].upper()}"
