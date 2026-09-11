"""Typed failures for business tools.

A tool never reports a failure as a normal string: every failure carries a
stable ``code`` and an explicit ``retryable`` flag, so the agent layer can decide
whether to retry, ask for more information, or tell the user it cannot help.
"""

from __future__ import annotations


class ToolError(Exception):
    """Base class for tool failures."""

    code: str = "tool_error"
    retryable: bool = False

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(ToolError):
    """The requested entity does not exist in the simulated data."""

    code = "not_found"


class PermissionDeniedError(ToolError):
    """The caller may not access this entity."""

    code = "permission_denied"


class InvalidArgumentError(ToolError):
    """The arguments failed schema or business validation."""

    code = "invalid_argument"


class ConfirmationRequiredError(ToolError):
    """A write operation was attempted without explicit user confirmation."""

    code = "confirmation_required"


class ConflictError(ToolError):
    """The write collided with existing state in a way retrying cannot fix."""

    code = "conflict"


class ToolUnavailableError(ToolError):
    """The backing store or dependency is temporarily unavailable."""

    code = "unavailable"
    retryable = True
