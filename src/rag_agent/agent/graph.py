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

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from rag_agent.agent.intent import Intent, classify_intent
from rag_agent.agent.nodes import DEFAULT_MAX_CLARIFICATIONS, AgentNodes
from rag_agent.agent.routing import prefer_knowledge_for_stream
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
from rag_agent.generation.rag_answer import (
    DEFAULT_MAX_CONTEXT_CHARS,
    RagAnswer,
    finalize_answer_with_repair,
    prepare_answer,
    stream_raw_answer,
)
from rag_agent.memory.conversation import (
    DEFAULT_WINDOW_SIZE,
    ConversationWindow,
    render_prompt_context,
    trim_messages,
)
from rag_agent.observability.metrics import StageTiming, TurnMetrics
from rag_agent.providers.base import ChatModel
from rag_agent.retrieval.retriever import Retriever
from rag_agent.storage.business import BusinessRepository
from rag_agent.tools.permissions import ToolPermissions

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
    injection_suspected: tuple[str, ...] = ()
    """Instruction-shaped text found in the retrieved evidence, if any.

    Reported so a reader can see that a document looked like it was giving orders; it never
    altered the answer, which still had to survive citation validation.
    """

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
            "injection_suspected": list(self.injection_suspected),
            "trace": list(self.trace),
        }


@dataclass(frozen=True, slots=True)
class _Contribution:
    """Channel lengths captured before a turn, used to slice out that turn's writes."""

    trace_before: int
    tools_before: int


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """One turn of a stored conversation, including the memory it used.

    ``tool_results`` and ``trace`` on ``run`` are **accumulated over the whole thread**,
    because that is what the checkpointer stores. Callers that show activity — the CLI's
    reader and the UI alike — need "what happened in *this* turn?" separately, which is
    what :attr:`turn_trace` and :attr:`turn_tool_results` provide: they are computed from
    the channel lengths captured before and after the invoke, so a resumed turn reports
    only what the resume actually did.
    """

    thread_id: str
    run: AgentRun
    window: ConversationWindow
    preferences: dict[str, str]
    pending_action: dict[str, Any] | None
    checkpoints_before: int
    checkpoints_after: int
    contribution: _Contribution | None = None
    metrics: TurnMetrics | None = None

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

    @property
    def turn_trace(self) -> tuple[str, ...]:
        """Only the trace entries this turn added."""

        if self.contribution is None:
            return self.run.trace
        return self.run.trace[self.contribution.trace_before :]

    @property
    def turn_tool_results(self) -> tuple[dict[str, Any], ...]:
        """Only the tool results this turn added."""

        if self.contribution is None:
            return self.run.tool_results
        return self.run.tool_results[self.contribution.tools_before :]

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
            "turn": {
                "trace": list(self.turn_trace),
                "tool_results": list(self.turn_tool_results),
            },
            "metrics": self.metrics.as_dict() if self.metrics is not None else None,
        }


def build_agent_graph(
    *,
    retriever: Retriever,
    chat_model: ChatModel,
    repository: BusinessRepository,
    max_clarifications: int = DEFAULT_MAX_CLARIFICATIONS,
    checkpointer: Any | None = None,
    permissions: ToolPermissions | None = None,
) -> Any:
    """Compile the workflow; node order and edges are fixed here, not by prompts.

    ``permissions`` is the caller's declared identity and role. It is a construction-time
    argument rather than a piece of graph state on purpose: nothing that flows through the
    model's reach can widen it, and the tool layer refuses when it is absent.
    """

    nodes = AgentNodes(
        retriever=retriever,
        chat_model=chat_model,
        repository=repository,
        max_clarifications=max_clarifications,
        permissions=permissions,
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
    permissions: ToolPermissions | None = None,
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
        permissions=permissions,
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
    contribution_before = _capture_contribution(graph, config)
    started = time.perf_counter()

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
        contribution=contribution_before,
        metrics=build_turn_metrics(
            cast("AgentState", result),
            thread_id=thread_id,
            total_ms=(time.perf_counter() - started) * 1000.0,
        ),
    )


def build_turn_metrics(
    state: AgentState,
    *,
    thread_id: str,
    total_ms: float,
) -> TurnMetrics:
    """Summarise what one turn did, from what the state actually recorded.

    Only measured values are reported. Two points worth being explicit about:

    * token counts come from the provider layer, which logs them but does not put them in the
      graph state, so this layer cannot supply them. They are reported as zero rather than
      estimated, and the CLI surfaces the provider's own token line alongside.
    * an attempt count above one means the provider retried. That is the closest thing to a
      per-stage signal available here without adding a tracing dependency.
    """

    metrics = TurnMetrics(thread_id=thread_id)
    metrics.stages.append(StageTiming(name="turn", duration_ms=total_ms))

    evidence_count = state.get("evidence_count")
    if isinstance(evidence_count, int):
        confident = state.get("retrieval_confident")
        metrics.record_retrieval(
            hits=evidence_count,
            confident=confident if isinstance(confident, bool) else None,
        )
    elif str(state.get("intent", "")) == str(Intent.KNOWLEDGE):
        # 走知识分支却没有证据数：检索为空，这是真实结果，不是「没测到」。
        metrics.record_retrieval(hits=0, confident=False)

    for entry in state.get("tool_results") or []:
        if not isinstance(entry, dict):
            continue
        ok = entry.get("status") == "ok"
        metrics.record_tool_result(
            tool=str(entry.get("tool", "unknown")),
            ok=ok,
            error_code=None if ok else str(entry.get("error_code") or ""),
        )
    return metrics


def _capture_contribution(graph: Any, config: dict[str, Any]) -> _Contribution:
    """Record how much accumulated output already existed before this turn.

    Read from the checkpoint rather than from the arguments, so it is correct on the
    resume path too, where the accumulated trace comes back with the restored state
    instead of from a fresh initial state.
    """

    snapshot = graph.get_state(config)
    values = snapshot.values if snapshot and snapshot.values else {}
    trace = values.get("trace") or []
    tools = values.get("tool_results") or []
    return _Contribution(
        trace_before=len(trace) if isinstance(trace, list) else 0,
        tools_before=len(tools) if isinstance(tools, list) else 0,
    )


def _count_checkpoints(graph: Any, thread_id: str) -> int:
    config = {"configurable": {"thread_id": thread_id}}
    return sum(1 for _ in graph.get_state_history(config))


def stream_chat_turn(
    graph: Any,
    question: str,
    *,
    retriever: Retriever,
    chat_model: ChatModel,
    thread_id: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    preferences: dict[str, str] | None = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    delegate_to_agent: bool = False,
    temperature: float = 0.0,
) -> Iterator[tuple[str, ChatTurn | None]]:
    """Run a knowledge turn with the thread's memory, streaming as it goes.

    ``chat_turn`` has to run the whole graph before it can report anything, which is
    fine for a CLI but wrong for a UI: the user would watch nothing happen. This entry
    point builds the *same* prompt — conversation window plus preferences plus numbered
    evidence, validated by the same citation checks — and yields the raw deltas while
    they arrive, then yields the finished :class:`ChatTurn` exactly once.

    The contract is "zero or more chunks, then exactly one turn", so a caller can forward
    the chunks to a streaming widget and still display the validated result. That is what
    makes streaming and citation validation compatible rather than a trade-off.

    ``retriever`` and ``chat_model`` are passed explicitly instead of being recovered
    from the compiled graph, so nothing here depends on LangGraph's node internals. The
    model output is the grounded JSON, so chunks exist for progress display only: callers
    render ``turn.run.answer``, never the raw stream.

    This path answers from the knowledge base only: it does not run the classifier, so it
    cannot route to the device tools or the ticket flow. ``delegate_to_agent`` makes that
    limit actionable instead of silent — when intent classification is available and the
    question is not a knowledge question, the turn is returned with
    ``intent_source == "delegated"`` and an empty answer, so the caller can hand the
    question to the full agent rather than showing a refusal that looks like a missing
    document. ``intent`` is left as ``device``/``ticket`` so the caller knows where to send
    it; a wrong guess degrades to an ordinary clarification, never to a lost question.
    """

    preferences = preferences or {}
    config = {"configurable": {"thread_id": thread_id}}
    window = ConversationWindow(
        messages=trim_messages(_history(graph, config), window_size=window_size)
    )
    prompt_context = render_prompt_context(window, preferences)
    checkpoints_before = _count_checkpoints(graph, thread_id)

    if delegate_to_agent:
        decision = classify_intent(question, chat_model=chat_model, context=prompt_context)
        if not prefer_knowledge_for_stream(str(decision.intent)):
            yield (
                "",
                _delegated_turn(
                    thread_id,
                    question,
                    window,
                    preferences,
                    str(decision.intent),
                    checkpoints_before,
                ),
            )
            return

    prepared = prepare_answer(question, retriever=retriever, max_context_chars=max_context_chars)
    if prepared.refusal is not None:
        yield "", _stream_turn(thread_id, window, preferences, prepared.refusal, checkpoints_before)
        return

    started = time.perf_counter()
    deltas: list[str] = []
    for delta in stream_raw_answer(
        prepared,
        chat_model,
        conversation_context=prompt_context,
        temperature=temperature,
    ):
        deltas.append(delta)
        yield delta, None

    answer = finalize_answer_with_repair(
        prepared,
        "".join(deltas),
        chat_model=chat_model,
        model=chat_model.model_name,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        temperature=temperature,
    )
    yield "", _stream_turn(thread_id, window, preferences, answer, checkpoints_before)


def _delegated_turn(
    thread_id: str,
    question: str,
    window: ConversationWindow,
    preferences: dict[str, str],
    intent: str,
    checkpoints_before: int,
) -> ChatTurn:
    """A turn that the knowledge-only stream declines to answer, for the caller to route.

    The answer is deliberately empty and ``intent_source`` is ``"delegated"``. Showing a
    refusal here would be wrong twice over: the question is not unanswerable, it just needs
    a branch this path does not have, and a refusal reads as "the documents do not cover
    it" — which is what the running UI showed for a simple device lookup.
    """

    return ChatTurn(
        thread_id=thread_id,
        run=AgentRun(
            question=question,
            status=STATUS_REFUSED,
            intent=intent,
            intent_confidence=0.0,
            intent_source="delegated",
            answer="",
            citations=(),
            trace=(f"stream:delegated:{intent}",),
            tool_results=(),
            pending_confirmation=None,
            created_ticket=None,
            error_code=None,
            error_message="",
            clarification_turns=0,
        ),
        window=window,
        preferences=dict(preferences),
        pending_action=None,
        checkpoints_before=checkpoints_before,
        checkpoints_after=checkpoints_before,
    )


def _stream_turn(
    thread_id: str,
    window: ConversationWindow,
    preferences: dict[str, str],
    answer: RagAnswer,
    checkpoints_before: int,
) -> ChatTurn:
    """Wrap a validated answer in the same result shape the graph nodes produce."""

    status = STATUS_ANSWERED if answer.answered else STATUS_REFUSED
    return ChatTurn(
        thread_id=thread_id,
        run=AgentRun(
            question=answer.question,
            status=status,
            intent=str(Intent.KNOWLEDGE),
            intent_confidence=0.0,
            intent_source="stream",
            answer=answer.text,
            citations=tuple(citation.as_dict() for citation in answer.citations),
            trace=(f"knowledge:{status}:citations={len(answer.citations)}",),
            tool_results=(),
            pending_confirmation=None,
            created_ticket=None,
            error_code=None if answer.refusal_cause is None else str(answer.refusal_cause),
            error_message=answer.reason,
            clarification_turns=0,
            injection_suspected=answer.injection_suspected,
        ),
        window=window,
        preferences=dict(preferences),
        pending_action=None,
        checkpoints_before=checkpoints_before,
        checkpoints_after=checkpoints_before,  # 流式路径只读不写检查点
    )


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
        injection_suspected=tuple(state.get("injection_suspected", ())),
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
