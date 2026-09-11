"""Graph nodes.

Each node does one thing and returns only the state keys it changed. Nodes never
decide *where* the graph goes next: that is expressed by the edges in
:mod:`rag_agent.agent.graph`, which is what keeps the business order — especially
the confirmation step before a write — out of reach of the model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langgraph.types import interrupt

from rag_agent.agent.intent import classify_intent
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
from rag_agent.generation.rag_answer import answer_with_context, extract_json_object
from rag_agent.providers.base import ChatMessage, ChatModel
from rag_agent.retrieval.retriever import Retriever
from rag_agent.storage.business import BusinessRepository
from rag_agent.tools.business import create_ticket, device_lookup, order_lookup
from rag_agent.tools.errors import ToolError
from rag_agent.tools.models import CreateTicketArgs, DeviceLookupArgs, OrderLookupArgs

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_CLARIFICATIONS = 3

#: Matches a device label such as ``D2002`` in a user message or in the recent
#: conversation. Deliberately strict: an uppercase ``D`` followed by digits only.
#: The simulated ids are ``D2001`` to ``D2005``, and no model name in the knowledge
#: base has this shape, so a false positive would have to be invented by the user.
DEVICE_ID_PATTERN = re.compile(r"\bD\d{3,}\b")

TICKET_EXTRACT_SYSTEM_PROMPT = (
    "从用户消息和已知信息中提取报修工单需要的字段。\n"
    "只输出一个 JSON 对象，格式为 "
    '{"device_id": string|null, "issue": string|null, "contact": string|null}，'
    "缺失的字段填 null，不要编造。"
)

CLARIFY_QUESTIONS = {
    "user_id": "请提供你的用户编号（例如 U1001）。",
    "device_id": "请提供设备编号（例如 D2001）。",
    "issue": "请描述具体故障现象，至少五个字。",
    "contact": "请提供联系方式，便于售后联系你。",
}

MISSING_FIELD_LABELS = {
    "user_id": "用户编号",
    "device_id": "设备编号",
    "issue": "故障描述",
    "contact": "联系方式",
}


@dataclass
class AgentNodes:
    """Node implementations bound to their dependencies."""

    retriever: Retriever
    chat_model: ChatModel
    repository: BusinessRepository
    max_clarifications: int = DEFAULT_MAX_CLARIFICATIONS

    def classify(self, state: AgentState) -> dict[str, object]:
        """Decide the task type; unusable answers fall back to ``unknown``."""

        decision = classify_intent(
            state.get("question", ""),
            chat_model=self.chat_model,
            context=state.get("prompt_context", ""),
        )
        return {
            "intent": str(decision.intent),
            "intent_confidence": decision.confidence,
            "intent_source": decision.source,
            "intent_reason": decision.reason,
            "trace": [f"classify:{decision.intent}"],
        }

    def knowledge(self, state: AgentState) -> dict[str, object]:
        """Answer from the knowledge base, with citations or an explicit refusal."""

        question = state.get("question", "")
        answer = answer_with_context(
            question,
            retriever=self.retriever,
            chat_model=self.chat_model,
            conversation_context=state.get("prompt_context", ""),
        )
        citations = [citation.as_dict() for citation in answer.citations]
        if answer.answered:
            return {
                "status": STATUS_ANSWERED,
                "answer": answer.text,
                "citations": citations,
                "evidence_count": len(answer.hits),
                "retrieval_confident": answer.confident,
                "trace": [f"knowledge:answered:citations={len(citations)}"],
            }
        return {
            "status": STATUS_REFUSED,
            "answer": "",
            "citations": [],
            "evidence_count": len(answer.hits),
            "retrieval_confident": answer.confident,
            "error_code": None if answer.refusal_cause is None else str(answer.refusal_cause),
            "error_message": answer.reason,
            "trace": [f"knowledge:refused:{answer.refusal_cause}"],
        }

    def device(self, state: AgentState) -> dict[str, object]:
        """Look up the caller's devices and orders; the store is the only source."""

        user_id = (state.get("user_id") or "").strip()
        if not user_id:
            return self._needs_input(state, ["user_id"], stage="device")
        requested_device = resolve_device_id(state)

        results: list[dict[str, object]] = []
        device_result = device_lookup(
            DeviceLookupArgs(device_id=requested_device, user_id=user_id),
            repository=self.repository,
        )
        results.append(device_result.model_dump())
        if not device_result.ok:
            return {
                "status": STATUS_ERROR,
                "tool_results": results,
                "tool_error_code": device_result.error_code,
                "error_code": device_result.error_code,
                "error_message": device_result.error_message,
                "trace": [f"device:failed:{device_result.error_code}"],
            }

        order_result = order_lookup(OrderLookupArgs(user_id=user_id), repository=self.repository)
        results.append(order_result.model_dump())
        return {
            "status": STATUS_ANSWERED,
            "answer": _summarize_device(device_result.data or {}, order_result.data),
            "tool_results": results,
            "tool_error_code": None,
            "trace": ["device:ok"],
        }

    def ticket_collect(self, state: AgentState) -> dict[str, object]:
        """Collect what a ticket needs, asking the model only to read the message."""

        extracted = self._extract_ticket_fields(state)
        draft: dict[str, object] = {
            "user_id": (state.get("user_id") or "").strip() or None,
            "device_id": (state.get("device_id") or extracted.get("device_id") or "").strip()
            or None,
            "issue": extracted.get("issue"),
            "contact": (state.get("contact") or extracted.get("contact") or "").strip() or None,
        }
        missing = [name for name in ("user_id", "device_id", "issue", "contact") if not draft[name]]
        if missing:
            return self._needs_input(state, missing, stage="ticket", draft=draft)
        return {
            "ticket_draft": draft,
            "missing_fields": [],
            "trace": ["ticket_collect:complete"],
        }

    def ticket_pending(self, state: AgentState) -> dict[str, object]:
        """Show the draft and stop. No write happens here."""

        draft = state.get("ticket_draft") or {}
        summary = _ticket_summary(draft)
        return {
            "status": STATUS_PENDING_CONFIRMATION,
            "answer": summary,
            "pending_confirmation": {
                "action": "ticket.create",
                "summary": summary,
                "draft": draft,
                "confirmed": False,
            },
            "trace": ["ticket_pending:awaiting_confirmation"],
        }

    def ticket_create(self, state: AgentState) -> dict[str, object]:
        """Ask the human to confirm, then write — in that order, every time.

        ``interrupt()`` is the *first* statement of this node, before any tool
        call. LangGraph raises it on the first execution and hands the run back to
        the caller with the graph paused here; the node is re-entered with the
        resume payload when the caller answers. Because the pause happens before
        the write, a cancellation cannot leave a partial ticket behind, and a
        resumed-and-then-failed run cannot double-write.
        """

        draft = state.get("ticket_draft") or {}
        decision = interrupt(
            {
                "action": "ticket.create",
                "summary": _ticket_summary(draft),
                "draft": draft,
                "instructions": (
                    "调用方传入 {'confirmed': true} 才写库，{'confirmed': false} 表示取消。"
                ),
            }
        )
        if not _is_confirmed(decision):
            return {
                "status": STATUS_TICKET_CANCELLED,
                "answer": "已取消，未创建工单。如需报修请重新发起请求。",
                "pending_confirmation": None,
                "created_ticket": None,
                "trace": ["ticket_create:cancelled"],
            }

        try:
            result = create_ticket(
                CreateTicketArgs(
                    user_id=str(draft.get("user_id", "")),
                    device_id=str(draft.get("device_id", "")),
                    issue=str(draft.get("issue", "")),
                    contact=str(draft.get("contact", "")),
                    confirmed=True,
                ),
                repository=self.repository,
            )
        except ToolError as exc:  # pragma: no cover - 工具内部已转成结果
            return {
                "status": STATUS_ERROR,
                "error_code": exc.code,
                "error_message": exc.message,
                "trace": [f"ticket_create:failed:{exc.code}"],
            }

        if not result.ok:
            return {
                "status": STATUS_ERROR,
                "tool_results": [result.model_dump()],
                "error_code": result.error_code,
                "error_message": result.error_message,
                "trace": [f"ticket_create:failed:{result.error_code}"],
            }
        data = result.data or {}
        ticket = data.get("ticket") or {}
        return {
            "status": STATUS_TICKET_CREATED,
            "created_ticket": ticket if isinstance(ticket, dict) else None,
            "pending_confirmation": None,
            "tool_results": [result.model_dump()],
            "answer": f"已创建售后工单 {ticket.get('ticket_id', '')}".strip(),
            "trace": [f"ticket_create:created={data.get('created')}"],
        }

    def clarify(self, state: AgentState) -> dict[str, object]:
        """Ask for the missing input instead of guessing."""

        turns = int(state.get("clarification_turns", 0)) + 1
        if turns > self.max_clarifications:
            return {
                "status": STATUS_ERROR,
                "clarification_turns": turns,
                "error_code": "clarification_limit",
                "error_message": (
                    f"clarification limit of {self.max_clarifications} reached; "
                    "handing over to a human agent"
                ),
                "trace": ["clarify:limit"],
            }

        missing = state.get("missing_fields") or []
        questions = [CLARIFY_QUESTIONS.get(name, f"请补充 {name}。") for name in missing]
        labels = "、".join(MISSING_FIELD_LABELS.get(name, name) for name in missing) or "更多信息"
        return {
            "status": STATUS_NEEDS_INPUT,
            "clarification_turns": turns,
            "answer": f"还需要你补充：{labels}。\n" + "\n".join(questions),
            "trace": [f"clarify:ask:{','.join(missing) or 'unknown'}"],
        }

    def fail(self, state: AgentState) -> dict[str, object]:
        """Terminal node for outcomes that cannot continue."""

        return {
            "status": state.get("status") or STATUS_ERROR,
            "trace": ["fail"],
        }

    def _needs_input(
        self,
        state: AgentState,
        missing: list[str],
        *,
        stage: str,
        draft: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "missing_fields": missing,
            "trace": [f"{stage}:missing:{','.join(missing)}"],
        }
        if draft is not None:
            payload["ticket_draft"] = draft
        return payload

    def _extract_ticket_fields(self, state: AgentState) -> dict[str, str | None]:
        """Read the ticket fields out of the message; the model is not asked to invent."""

        response = self.chat_model.chat(
            [
                ChatMessage(role="system", content=TICKET_EXTRACT_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=f"{state.get('prompt_context', '')}{state.get('question', '')}",
                ),
            ]
        )
        payload = extract_json_object(response.text) or {}
        return {
            "device_id": _as_text(payload.get("device_id")),
            "issue": _as_text(payload.get("issue")),
            "contact": _as_text(payload.get("contact")),
        }


def _is_confirmed(decision: object) -> bool:
    """Accept only an explicit confirmation from the resume payload."""

    if isinstance(decision, dict):
        return decision.get("confirmed") is True
    return False


def resolve_device_id(state: AgentState) -> str | None:
    """Find the device this turn is about, without asking the model.

    The value comes from the caller first, then from the text: the current message
    is the strongest signal, and the conversation window is what makes a follow-up
    such as "那它的保修期是多久" resolvable at all.

    This is deliberately a *deterministic* extraction instead of another model call.
    Before it existed, the device id could only arrive through the ``--device-id``
    flag, so asking "D2002 还在保修吗" in a message listed every device instead of
    answering about D2002 — and once a previous turn was rendered into the prompt,
    the same follow-up was classified as a knowledge question often enough to turn
    the whole flow into a refusal.

    Ownership is *not* checked here. The extracted id is handed to the tool exactly
    like a caller-supplied one, so a device belonging to somebody else still fails
    with ``permission_denied`` instead of being silently skipped.
    """

    supplied = (state.get("device_id") or "").strip()
    if supplied:
        return supplied
    for text in (state.get("question", ""), state.get("prompt_context", "")):
        match = DEVICE_ID_PATTERN.search(text or "")
        if match:
            return match.group(0).upper()
    return None


def _as_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ticket_summary(draft: dict[str, object]) -> str:
    return (
        "即将提交的工单：\n"
        f"- 用户编号：{draft.get('user_id')}\n"
        f"- 设备编号：{draft.get('device_id')}\n"
        f"- 故障描述：{draft.get('issue')}\n"
        f"- 联系方式：{draft.get('contact')}\n"
        "确认后才会创建工单，请回复“确认”或“取消”。"
    )


def _summarize_device(device_payload: dict[str, object], orders: dict[str, object] | None) -> str:
    lines: list[str] = []
    device = device_payload.get("device")
    if isinstance(device, dict):
        status = "在保" if device.get("warranty_active") else "已过保"
        lines.append(
            f"设备 {device.get('device_id')}（{device.get('model')}）{status}，"
            f"保修到期日 {device.get('warranty_expires_on')}。"
        )
    devices = device_payload.get("devices")
    if isinstance(devices, list):
        lines.append(f"该用户共有 {len(devices)} 台设备。")
        for item in devices:
            if isinstance(item, dict):
                status = "在保" if item.get("warranty_active") else "已过保"
                lines.append(f"- {item.get('device_id')}（{item.get('model')}）{status}")
    if orders is not None:
        count = orders.get("count", 0)
        lines.append(f"关联订单 {count} 笔。")
    return "\n".join(lines) if lines else "未查询到设备信息。"
