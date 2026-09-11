"""The agent graph: deterministic routing around model-assisted decisions.

The model classifies intent and reads fields out of a message. Everything that
must happen in a fixed order — looking up business data, showing a draft, and
requiring confirmation before a write — is an edge, not a prompt instruction.

Two guards bound the run: an explicit clarification counter, and LangGraph's own
``recursion_limit``. There is intentionally no cycle in the graph today, because
without a checkpointer no new user input can arrive mid-run; stage 11 replaces
the clarification return with an Interrupt that resumes in a later turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from rag_agent.agent.intent import Intent
from rag_agent.agent.nodes import DEFAULT_MAX_CLARIFICATIONS, AgentNodes
from rag_agent.agent.state import (
    STATUS_ANSWERED,
    STATUS_ERROR,
    STATUS_NEEDS_INPUT,
    STATUS_PENDING_CONFIRMATION,
    STATUS_REFUSED,
    STATUS_TICKET_CREATED,
    AgentState,
)
from rag_agent.providers.base import ChatModel
from rag_agent.retrieval.retriever import Retriever
from rag_agent.storage.business import BusinessRepository

DEFAULT_RECURSION_LIMIT = 12

NODE_CLASSIFY = "classify"
NODE_KNOWLEDGE = "knowledge"
NODE_DEVICE = "device"
NODE_TICKET_COLLECT = "ticket_collect"
NODE_TICKET_PENDING = "ticket_pending"
NODE_TICKET_CREATE = "ticket_create"
NODE_CLARIFY = "clarify"
NODE_FAIL = "fail"


@dataclass(frozen=True, slots=True)
class AgentRun:
    """Typed view over the state a run finished with."""

    question: str
    status: str
    intent: str
    intent_confidence: float
    intent_source: str
    answer: str
    citations: tuple[dict[str, Any], ...]
    trace: tuple[str, ...]
    tool_results: tuple[dict[str, Any], ...]
    pending_confirmation: dict[str, Any] | None
    created_ticket: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    clarification_turns: int

    @property
    def answered(self) -> bool:
        return self.status == STATUS_ANSWERED

    def as_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "status": self.status,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "intent_source": self.intent_source,
            "answer": self.answer,
            "citations": list(self.citations),
            "tool_results": list(self.tool_results),
            "pending_confirmation": self.pending_confirmation,
            "created_ticket": self.created_ticket,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "clarification_turns": self.clarification_turns,
            "trace": list(self.trace),
        }


def build_agent_graph(
    *,
    retriever: Retriever,
    chat_model: ChatModel,
    repository: BusinessRepository,
    max_clarifications: int = DEFAULT_MAX_CLARIFICATIONS,
) -> Any:
    """Compile the workflow; node order and edges are fixed here, not by prompts."""

    nodes = AgentNodes(
        retriever=retriever,
        chat_model=chat_model,
        repository=repository,
        max_clarifications=max_clarifications,
    )

    builder: Any = StateGraph(AgentState)
    builder.add_node(NODE_CLASSIFY, nodes.classify)
    builder.add_node(NODE_KNOWLEDGE, nodes.knowledge)
    builder.add_node(NODE_DEVICE, nodes.device)
    builder.add_node(NODE_TICKET_COLLECT, nodes.ticket_collect)
    builder.add_node(NODE_TICKET_PENDING, nodes.ticket_pending)
    builder.add_node(NODE_TICKET_CREATE, nodes.ticket_create)
    builder.add_node(NODE_CLARIFY, nodes.clarify)
    builder.add_node(NODE_FAIL, nodes.fail)

    builder.add_edge(START, NODE_CLASSIFY)
    builder.add_conditional_edges(
        NODE_CLASSIFY,
        _route_by_intent,
        {
            NODE_KNOWLEDGE: NODE_KNOWLEDGE,
            NODE_DEVICE: NODE_DEVICE,
            NODE_TICKET_COLLECT: NODE_TICKET_COLLECT,
            NODE_CLARIFY: NODE_CLARIFY,
        },
    )
    builder.add_edge(NODE_KNOWLEDGE, END)
    builder.add_conditional_edges(
        NODE_DEVICE,
        _route_after_collection,
        {NODE_CLARIFY: NODE_CLARIFY, NODE_FAIL: NODE_FAIL, END: END},
    )
    builder.add_conditional_edges(
        NODE_TICKET_COLLECT,
        _route_after_ticket_collection,
        {NODE_TICKET_PENDING: NODE_TICKET_PENDING, NODE_CLARIFY: NODE_CLARIFY},
    )
    builder.add_conditional_edges(
        NODE_TICKET_PENDING,
        _route_after_pending,
        {NODE_TICKET_CREATE: NODE_TICKET_CREATE, END: END},
    )
    builder.add_edge(NODE_TICKET_CREATE, END)
    builder.add_edge(NODE_CLARIFY, END)
    builder.add_edge(NODE_FAIL, END)

    return builder.compile()


def run_agent(
    question: str,
    *,
    retriever: Retriever,
    chat_model: ChatModel,
    repository: BusinessRepository,
    user_id: str | None = None,
    device_id: str | None = None,
    contact: str | None = None,
    clarification_turns: int = 0,
    confirmation: dict[str, Any] | None = None,
    max_clarifications: int = DEFAULT_MAX_CLARIFICATIONS,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> AgentRun:
    """Run one turn of the workflow and return a typed result."""

    graph = build_agent_graph(
        retriever=retriever,
        chat_model=chat_model,
        repository=repository,
        max_clarifications=max_clarifications,
    )
    initial: AgentState = {
        "question": question,
        "user_id": user_id,
        "device_id": device_id,
        "contact": contact,
        "clarification_turns": clarification_turns,
        "confirmation": confirmation,
        "trace": [],
        "tool_results": [],
        "messages": [],
    }
    final: AgentState = graph.invoke(initial, config={"recursion_limit": recursion_limit})
    return _to_run(question, final)


def _to_run(question: str, state: AgentState) -> AgentRun:
    return AgentRun(
        question=question,
        status=str(state.get("status", STATUS_ERROR)),
        intent=str(state.get("intent", Intent.UNKNOWN)),
        intent_confidence=float(state.get("intent_confidence", 0.0)),
        intent_source=str(state.get("intent_source", "unknown")),
        answer=str(state.get("answer", "")),
        citations=tuple(state.get("citations", ())),
        trace=tuple(state.get("trace", ())),
        tool_results=tuple(state.get("tool_results", ())),
        pending_confirmation=state.get("pending_confirmation"),
        created_ticket=state.get("created_ticket"),
        error_code=state.get("error_code"),
        error_message=state.get("error_message"),
        clarification_turns=int(state.get("clarification_turns", 0)),
    )


def _route_by_intent(state: AgentState) -> str:
    intent = str(state.get("intent", Intent.UNKNOWN))
    if intent == Intent.KNOWLEDGE:
        return NODE_KNOWLEDGE
    if intent == Intent.DEVICE:
        return NODE_DEVICE
    if intent == Intent.TICKET:
        return NODE_TICKET_COLLECT
    return NODE_CLARIFY


def _route_after_collection(state: AgentState) -> str:
    status = str(state.get("status", ""))
    if status in (STATUS_ANSWERED, STATUS_REFUSED):
        return END
    if status == STATUS_ERROR:
        return NODE_FAIL
    return NODE_CLARIFY


def _route_after_ticket_collection(state: AgentState) -> str:
    return NODE_TICKET_PENDING if not state.get("missing_fields") else NODE_CLARIFY


def _route_after_pending(state: AgentState) -> str:
    confirmation = state.get("confirmation") or {}
    if confirmation.get("confirmed") is True:
        return NODE_TICKET_CREATE
    return END


__all__ = [
    "DEFAULT_RECURSION_LIMIT",
    "NODE_CLARIFY",
    "NODE_CLASSIFY",
    "NODE_DEVICE",
    "NODE_FAIL",
    "NODE_KNOWLEDGE",
    "NODE_TICKET_COLLECT",
    "NODE_TICKET_CREATE",
    "NODE_TICKET_PENDING",
    "STATUS_ANSWERED",
    "STATUS_ERROR",
    "STATUS_NEEDS_INPUT",
    "STATUS_PENDING_CONFIRMATION",
    "STATUS_REFUSED",
    "STATUS_TICKET_CREATED",
    "AgentRun",
    "build_agent_graph",
    "run_agent",
]
