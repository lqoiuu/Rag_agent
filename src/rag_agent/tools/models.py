"""Pydantic schemas for tool arguments and results.

Schemas are the contract: the model may propose arguments, but only values that
validate here reach the repository, and every result keeps the tool name, a
request id and an explicit status so callers never parse prose.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ToolStatus = Literal["ok", "error"]

MIN_ISSUE_CHARS = 5
MIN_CONTACT_CHARS = 3


class ToolResult(BaseModel):
    """Uniform tool reply consumed by the CLI and, later, by the agent graph."""

    tool: str
    status: ToolStatus
    request_id: str
    data: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class UserLookupArgs(BaseModel):
    """Look up one simulated user."""

    user_id: str = Field(min_length=1)


class DeviceLookupArgs(BaseModel):
    """Look up devices by device id, by owning user, or both."""

    device_id: str | None = Field(default=None, min_length=1)
    user_id: str | None = Field(default=None, min_length=1)
    as_of: date | None = None

    @model_validator(mode="after")
    def require_a_selector(self) -> DeviceLookupArgs:
        if self.device_id is None and self.user_id is None:
            raise ValueError("device_id or user_id is required")
        return self


class OrderLookupArgs(BaseModel):
    """Look up orders by order id or by owning user."""

    order_id: str | None = Field(default=None, min_length=1)
    user_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_a_selector(self) -> OrderLookupArgs:
        if self.order_id is None and self.user_id is None:
            raise ValueError("order_id or user_id is required")
        return self


class TicketDraft(BaseModel):
    """Validated content of a pending ticket."""

    user_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    issue: str = Field(min_length=MIN_ISSUE_CHARS, max_length=500)
    contact: str = Field(min_length=MIN_CONTACT_CHARS, max_length=120)

    def summary(self) -> str:
        return f"{self.device_id} / {self.issue} / {self.contact}"

    def idempotency_key(self) -> str:
        """Derive a stable key so the same draft never creates two tickets."""

        digest = hashlib.sha256(
            "|".join(
                (
                    self.user_id.strip(),
                    self.device_id.strip(),
                    " ".join(self.issue.split()),
                    " ".join(self.contact.split()),
                )
            ).encode("utf-8")
        ).hexdigest()
        return f"idem-{digest[:24]}"


class CreateTicketArgs(BaseModel):
    """Arguments of the write tool, including the confirmation contract.

    ``confirmed`` must be set by the caller only after a user has reviewed the
    draft. The tool refuses to write while it is false, so a model cannot create
    a ticket on its own.
    """

    user_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    issue: str = Field(min_length=MIN_ISSUE_CHARS, max_length=500)
    contact: str = Field(min_length=MIN_CONTACT_CHARS, max_length=120)
    confirmed: bool = False
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=64)

    def draft(self) -> TicketDraft:
        return TicketDraft(
            user_id=self.user_id,
            device_id=self.device_id,
            issue=self.issue,
            contact=self.contact,
        )
