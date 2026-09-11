"""Business tools with explicit contracts and typed failures."""

from rag_agent.tools.business import (
    DEVICE_LOOKUP,
    ORDER_LOOKUP,
    TICKET_CREATE,
    TOOL_NAMES,
    USER_LOOKUP,
    create_ticket,
    device_lookup,
    order_lookup,
    user_lookup,
)
from rag_agent.tools.errors import (
    ConfirmationRequiredError,
    ConflictError,
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
    TicketDraft,
    ToolResult,
    UserLookupArgs,
)

__all__ = [
    "DEVICE_LOOKUP",
    "ORDER_LOOKUP",
    "TICKET_CREATE",
    "TOOL_NAMES",
    "USER_LOOKUP",
    "ConfirmationRequiredError",
    "ConflictError",
    "CreateTicketArgs",
    "DeviceLookupArgs",
    "InvalidArgumentError",
    "NotFoundError",
    "OrderLookupArgs",
    "PermissionDeniedError",
    "TicketDraft",
    "ToolError",
    "ToolResult",
    "ToolUnavailableError",
    "UserLookupArgs",
    "create_ticket",
    "device_lookup",
    "order_lookup",
    "user_lookup",
]
