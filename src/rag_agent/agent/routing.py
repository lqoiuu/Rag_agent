"""Deterministic routing corrections applied after the model classifies.

The model decides intent, but one of its decisions proved unstable in a way that
breaks a whole flow: the *same* follow-up question was classified as a device
question when the conversation was empty and as a knowledge question once one turn
of history was rendered into the prompt (3/3 in both directions, measured with the
real model). The knowledge branch then refused, because nothing in the knowledge
base answers "how long is its warranty".

This module is where that is corrected. It is deliberately tiny and pure:

* it can only ever move a classification **towards** ``device``;
* it requires a device label that is already resolvable from the text, so it cannot
  invent a device that was never mentioned;
* it fires when the message either points back at the previous turn or asks about a
  field that exists only in the device store, and it stays out of the way when the
  message names a product-knowledge topic.

Everything here is a plain function over strings, so the boundaries can be tested by
enumeration instead of hope.
"""

from __future__ import annotations

from rag_agent.agent.devices import DEVICE_ID_PATTERN

#: Intent labels that may be replaced. ``device`` needs no correction, and ``ticket``
#: is an explicit action request that must never be redirected: turning a repair
#: request into a status lookup would be worse than misclassifying it.
REPLACEABLE_INTENTS = ("knowledge", "unknown")

#: Words that mark a follow-up as pointing back at something already discussed.
REFERENTIAL_CUES = (
    "它",
    "这个",
    "那个",
    "该设备",
    "该产品",
    "这台",
    "上述",
    "上面",
    "同上",
    "呢？",
    "呢?",
)

#: Fields that exist only in the device store, never in the product manual. A question
#: about one of these is a device-record question even when the device is named outright
#: instead of referred to as "it".
#:
#: This list is what makes the correction fire in the commonest case. The rule originally
#: required a *referential* follow-up, so "D2002 还在保修吗" -- a plainly named device with
#: an unambiguous record question -- was still misrouted to the knowledge branch whenever
#: the model classified it as knowledge, and the user got a refusal they could not act on.
#: That is exactly what happened in the running UI.
DEVICE_RECORD_CUES = (
    "保修",
    "订单",
    "购买",
    "下单",
    "激活",
    "到期",
    "过保",
    "在保",
    "序列号",
    "我的设备",
    "我的机器",
)

#: Words that mean the question is about product knowledge rather than about this
#: device's records. Their presence cancels the correction, so "那它的耗电量是多少"
#: stays a knowledge question instead of becoming a device lookup. The cost is a
#: miss, never a hijack.
#:
#: Warranty and duration words are deliberately *absent* from this list: "还在保修吗",
#: "保修期到多长时间" and "订单还有多久" are answerable only from the device store, so
#: treating them as knowledge topics would suppress the very case this correction exists
#: for. They live in :data:`DEVICE_RECORD_CUES` instead.
KNOWLEDGE_TOPIC_WORDS = (
    "型号",
    "参数",
    "规格",
    "功率",
    "耗电",
    "充电",
    "电压",
    "额定",
    "怎么",
    "如何",
    "为什么",
    "为何",
    "故障",
    "报错",
    "保养",
    "清洁",
    "清洗",
    "更换",
    "滤芯",
    "滤网",
    "刷",
    "噪音",
    "安全",
    "注意",
    "说明",
    "标准",
    "尺寸",
    "重量",
    "材质",
)


def is_referential(question: str) -> bool:
    """True when the message points back at something already discussed."""

    return any(cue in question for cue in REFERENTIAL_CUES)


def asks_about_device_records(question: str) -> bool:
    """True when the message asks about a field only the device store holds."""

    return any(cue in question for cue in DEVICE_RECORD_CUES)


def mentions_knowledge_topic(question: str) -> bool:
    """True when the message is about product knowledge rather than device records.

    The device label is removed first, so "D2002 的保修期" is judged on "的保修期".
    """

    stripped = DEVICE_ID_PATTERN.sub(" ", question)
    return any(word in stripped for word in KNOWLEDGE_TOPIC_WORDS)


#: Intents the streaming knowledge path can actually answer. Anything else must be handed
#: to the full agent, because the streaming path does not run the classifier and therefore
#: has no way to reach the device tools or the ticket flow.
STREAMABLE_INTENTS = ("knowledge", "unknown")


def prefer_knowledge_for_stream(intent: str) -> bool:
    """Whether the knowledge-only streaming path should handle this question itself.

    ``unknown`` stays here on purpose: an unclassifiable question is still worth a
    grounded attempt, and a genuinely unanswerable one is refused by the citation checks
    anyway.
    """

    return intent in STREAMABLE_INTENTS


def prefer_device(question: str, device_id: str | None, intent: str) -> bool:
    """Whether a model classification should be corrected to ``device``.

    Each condition guards against a different mistake:

    * the intent must be replaceable -- never ``ticket``;
    * a device id must already be resolvable -- never invent one;
    * the message must be about this device: it either points back at the previous turn
      or asks about a device-record field, so a message that merely happens to mention a
      device label is not captured;
    * the message must not name a product-knowledge topic -- otherwise a real knowledge
      question gets hijacked into a device listing.
    """

    if intent not in REPLACEABLE_INTENTS:
        return False
    if not device_id:
        return False
    if not (is_referential(question) or asks_about_device_records(question)):
        return False
    return not mentions_knowledge_topic(question)


__all__ = [
    "DEVICE_RECORD_CUES",
    "KNOWLEDGE_TOPIC_WORDS",
    "REFERENTIAL_CUES",
    "REPLACEABLE_INTENTS",
    "STREAMABLE_INTENTS",
    "asks_about_device_records",
    "is_referential",
    "mentions_knowledge_topic",
    "prefer_device",
    "prefer_knowledge_for_stream",
]
