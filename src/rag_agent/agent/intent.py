"""Model-assisted intent classification with a deterministic fallback.

The model may *propose* an intent, but anything unusable — malformed JSON, an
unknown label, or confidence below the threshold — collapses to
:attr:`Intent.UNKNOWN`, which routes to a clarification branch. Routing itself
never depends on free text.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rag_agent.generation.rag_answer import extract_json_object
from rag_agent.providers.base import ChatMessage, ChatModel

DEFAULT_MIN_CONFIDENCE = 0.5

INTENT_SYSTEM_PROMPT = (
    "你是售后知识库系统的意图分类器。请判断用户请求属于哪一类，只能选择一个：\n"
    "- knowledge：询问产品知识、使用方法、故障处理等知识库问题；\n"
    "- device：查询特定用户或设备的业务数据，例如保修状态、订单信息；\n"
    "- ticket：要求报修或创建售后工单；\n"
    "- unknown：无法判断，或请求超出上述范围。\n"
    "只输出一个 JSON 对象，格式为 "
    '{"intent": "knowledge|device|ticket|unknown", "confidence": 0.0-1.0, "reason": "简述理由"}，'
    "不要输出任何其他文字。"
)


class Intent(StrEnum):
    """Task types the workflow can route."""

    KNOWLEDGE = "knowledge"
    DEVICE = "device"
    TICKET = "ticket"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class IntentDecision:
    """One classification result plus how it was obtained."""

    intent: Intent
    confidence: float
    source: str
    reason: str = ""

    @property
    def from_model(self) -> bool:
        return self.source == "model"


def classify_intent(
    question: str,
    *,
    chat_model: ChatModel,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    temperature: float = 0.0,
    context: str = "",
) -> IntentDecision:
    """Classify one question, falling back to ``unknown`` when unusable.

    ``context`` carries the recent conversation so a follow-up such as
    "那它还在保修吗" can be classified at all. It is prompt input only: routing
    still depends on the returned label and confidence, never on the context text.
    """

    cleaned = question.strip()
    if not cleaned:
        return _fallback("question is empty")

    response = chat_model.chat(
        [
            ChatMessage(role="system", content=INTENT_SYSTEM_PROMPT),
            ChatMessage(role="user", content=f"{context}{cleaned}"),
        ],
        temperature=temperature,
    )
    payload = extract_json_object(response.text)
    if payload is None:
        return _fallback("model did not return a JSON object")

    label = str(payload.get("intent", "")).strip().lower()
    try:
        intent = Intent(label)
    except ValueError:
        return _fallback(f"unrecognised intent label {label!r}")

    confidence = _as_confidence(payload.get("confidence"))
    reason = str(payload.get("reason", "")).strip()
    if confidence < min_confidence:
        return IntentDecision(
            intent=Intent.UNKNOWN,
            confidence=confidence,
            source="model",
            reason=f"confidence {confidence:.2f} is below {min_confidence:.2f}",
        )
    return IntentDecision(intent=intent, confidence=confidence, source="model", reason=reason)


def _fallback(reason: str) -> IntentDecision:
    return IntentDecision(intent=Intent.UNKNOWN, confidence=0.0, source="fallback", reason=reason)


def _as_confidence(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    if isinstance(value, str):
        try:
            return min(1.0, max(0.0, float(value.strip())))
        except ValueError:
            return 0.0
    return 0.0
