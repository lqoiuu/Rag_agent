"""The agent graph: deterministic routing around model-assisted decisions.

The model classifies intent and reads fields out of a message. Everything that
must happen in a fixed order — looking up business data, showing a draft, and
requiring confirmation before a write — is an edge, not a prompt instruction.

Three guards bound a run: an explicit clarification counter, LangGraph's own
``recursion_limit``, and the confirmation pause in :meth:`AgentNodes.ticket_create`.

With a checkpointer attached the same graph also serves multi-turn conversations:
``run_agent`` is the single-turn entry point used by tests and debugging, while
``chat_turn`` drives one turn of a stored thread, including resuming a paused
write. The graph itself is unchanged by memory — only the config and the input
type differ, which is why the confirmation pause is a node-level ``interrupt()``
rather than a change to the edge set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from rag_agent.agent.intent import Intent
from rag_agent.agent.nodes import DEFAULT_MAX_CLARIFICATIONS, AgentNodes
from rag_agent.agent.state import (
    STATUS_ANSWERED,
    STATUS_ERROR,
    STATUS_NEEDS_INPUT,
    STATUS_PENDING_CONFIRMATION,
    STATUS_REFUSED,
    STATUS_TICKET_CANCELLED,
    STATUS_TICKET_CREATED,
    AgentState,
)
from rag_agent.memory.conversation import (
    DEFAULT_WINDOW_SIZE,
    ConversationWindow,
    render_prompt_context,
    trim_messages,
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


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """One turn of a stored conversation, including the memory it used."""

    thread_id: str
    run: AgentRun
    window: ConversationWindow
    preferences: dict[str, str]
    pending_action: dict[str, Any] | None
    checkpoints_before: int
    checkpoints_after: int

    @property
    def status(self) -> str:
        return self.run.status

    @property
    def paused(self) -> bool:
        """True when the graph stopped on an interrupt and awaits a decision."""

        return self.pending_action is not None

    @property
    def answered(self) -> bool:
        return self.run.answered

    def as_state(self, *, user_id: str | None = None, question: str = "") -> dict[str, object]:
        """State to persist as a conversation message after this turn."""

        return {
            "thread_id": self.thread_id,
            "user_id": user_id,
            "question": question or self.run.question,
            "status": self.run.status,
            "intent": self.run.intent,
            "answer": self.run.answer,
            "citations": list(self.run.citations),
            "created_ticket": self.run.created_ticket,
            "pending_confirmation": self.run.pending_confirmation,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "thread_id": self.thread_id,
            **self.run.as_dict(),
            "paused": self.paused,
            "pending_action": self.pending_action,
            "memory": {
                "window_size": len(self.window.messages),
                "preferences": len(self.preferences),
                "checkpoints_before": self.checkpoints_before,
                "checkpoints_after": self.checkpoints_after,
            },
        }


def build_agent_graph(
    *,
    retriever: Retriever,
    chat_model: ChatModel,
    repository: BusinessRepository,
    max_clarifications: int = DEFAULT_MAX_CLARIFICATIONS,
    checkpointer: Any | None = None,
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
    # 待确认节点后是一条普通边：确认与否则由 ticket_create 内部的 interrupt 决定。
    # 若这里再判断一次 confirmation，图上就会多出一条「未确认即结束」的隐含路径，
    # 让暂停点从结构上消失。
    builder.add_edge(NODE_TICKET_PENDING, NODE_TICKET_CREATE)
    builder.add_edge(NODE_TICKET_CREATE, END)
    builder.add_edge(NODE_CLARIFY, END)
    builder.add_edge(NODE_FAIL, END)

    return builder.compile(checkpointer=checkpointer)


def build_initial_state(
    question: str,
    *,
    thread_id: str = "",
    user_id: str | None = None,
    device_id: str | None = None,
    contact: str | None = None,
    clarification_turns: int = 0,
    window: ConversationWindow | None = None,
    preferences: dict[str, str] | None = None,
) -> AgentState:
    """Assemble the state one turn starts from, including its prompt window."""

    conversation = window or ConversationWindow(messages=())
    return {
        "question": question,
        "thread_id": thread_id,
        "user_id": user_id,
        "device_id": device_id,
        "contact": contact,
        "clarification_turns": clarification_turns,
        "confirmation": None,
        "prompt_context": render_prompt_context(conversation, preferences),
        "preferences": dict(preferences or {}),
        "trace": [],
        "tool_results": [],
        "messages": [],
    }


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
    max_clarifications: int = DEFAULT_MAX_CLARIFICATIONS,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> AgentRun:
    """Run one stateless turn and return a typed result.

    Without a checkpointer there is nowhere to pause, so a run that needs human
    confirmation ends at the ``pending_confirmation`` status — it does not write.
    Use :func:`chat_turn` for the resumable, thread-scoped version.
    """

    graph = build_agent_graph(
        retriever=retriever,
        chat_model=chat_model,
        repository=repository,
        max_clarifications=max_clarifications,
    )
    initial = build_initial_state(
        question,
        user_id=user_id,
        device_id=device_id,
        contact=contact,
        clarification_turns=clarification_turns,
    )
    final: AgentState = graph.invoke(initial, config={"recursion_limit": recursion_limit})
    return _to_run(question, final)


def chat_turn(
    graph: Any,
    question: str,
    *,
    thread_id: str,
    user_id: str | None = None,
    device_id: str | None = None,
    contact: str | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    preferences: dict[str, str] | None = None,
    resume: dict[str, Any] | None = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> ChatTurn:
    """Drive one turn of a stored thread, resuming a paused write when asked.

    ``resume`` is the human decision for a pending action. It is delivered through
    ``Command(resume=...)`` so the node that paused receives it directly; nothing
    about the decision is written into the state by the caller.
    """

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    checkpoints_before = _count_checkpoints(graph, thread_id)

    if resume is None:
        history = _history(graph, config)
        window = ConversationWindow(messages=trim_messages(history, window_size=window_size))
        initial = build_initial_state(
            question,
            thread_id=thread_id,
            user_id=user_id,
            device_id=device_id,
            contact=contact,
            window=window,
            preferences=preferences,
        )
        result: dict[str, Any] = graph.invoke(initial, config=config)
    else:
        window = ConversationWindow(
            messages=trim_messages(_history(graph, config), window_size=window_size)
        )
        result = graph.invoke(Command(resume=resume), config=config)

    run = _to_run(question or str(result.get("question", "")), cast("AgentState", result))
    return ChatTurn(
        thread_id=thread_id,
        run=run,
        window=window,
        preferences=dict(preferences or {}),
        pending_action=_pending_action(graph, config, result),
        checkpoints_before=checkpoints_before,
        checkpoints_after=_count_checkpoints(graph, thread_id),
    )


def _count_checkpoints(graph: Any, thread_id: str) -> int:
    config = {"configurable": {"thread_id": thread_id}}
    return sum(1 for _ in graph.get_state_history(config))


def _history(graph: Any, config: dict[str, Any]) -> list[Any]:
    """Read the accumulated conversation messages out of the checkpoint.

    The raw list is returned untrimmed; the caller applies the window so the same
    helper serves both "what do we remember" and "what goes into the prompt".
    """

    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.values:
        return []
    raw = snapshot.values.get("messages", [])
    return list(raw) if isinstance(raw, list) else []


def _pending_action(
    graph: Any,
    config: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the interrupt payload when the graph is paused, else ``None``.

    The payload is read from the run result first and from the graph snapshot
    second, so a caller that only keeps the result still sees what is pending.
    """

    interrupts = result.get("__interrupt__")
    if not interrupts:
        snapshot = graph.get_state(config)
        tasks = getattr(snapshot, "tasks", ()) or ()
        interrupts = [item for task in tasks for item in getattr(task, "interrupts", ())]
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", None)
    return value if isinstance(value, dict) else {"action": str(value)}


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


__all__ = [
    "DEFAULT_RECURSION_LIMIT",
    "DEFAULT_WINDOW_SIZE",
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
    "STATUS_TICKET_CANCELLED",
    "STATUS_TICKET_CREATED",
    "AgentRun",
    "ChatTurn",
    "build_agent_graph",
    "build_initial_state",
    "chat_turn",
    "run_agent",
]
