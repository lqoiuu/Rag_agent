"""Graph state shared by every node.

The state is a ``TypedDict`` with explicit reducers for the lists that several
nodes append to, so a node returns only the keys it changed and the trace stays
additive. Statuses are plain strings with documented meanings rather than
implicit flags, which keeps the routing conditions readable.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from rag_agent.providers.base import ChatMessage

STATUS_ANSWERED = "answered"
STATUS_REFUSED = "refused"
STATUS_NEEDS_INPUT = "needs_input"
STATUS_PENDING_CONFIRMATION = "pending_confirmation"
STATUS_TICKET_CREATED = "ticket_created"
STATUS_ERROR = "error"

TERMINAL_STATUSES = (
    STATUS_ANSWERED,
    STATUS_REFUSED,
    STATUS_NEEDS_INPUT,
    STATUS_PENDING_CONFIRMATION,
    STATUS_TICKET_CREATED,
    STATUS_ERROR,
)


class AgentState(TypedDict, total=False):
    """Everything a node may read or write during one run."""

    # 输入
    question: str
    user_id: str | None
    device_id: str | None
    contact: str | None

    # 意图识别
    intent: str
    intent_confidence: float
    intent_source: str
    intent_reason: str

    # 澄清
    clarification_turns: int
    missing_fields: list[str]

    # 知识与工具结果
    messages: Annotated[list[ChatMessage], operator.add]
    answer: str
    citations: list[dict[str, object]]
    evidence_count: int
    retrieval_confident: bool
    tool_results: Annotated[list[dict[str, object]], operator.add]
    tool_error_code: str | None

    # 工单流程
    ticket_draft: dict[str, object] | None
    pending_confirmation: dict[str, object] | None
    confirmation: dict[str, object] | None
    created_ticket: dict[str, object] | None

    # 结果与可观测性
    status: str
    error_code: str | None
    error_message: str | None
    trace: Annotated[list[str], operator.add]
