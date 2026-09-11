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
* it stays out of the way unless the message is a *referential* follow-up, which is
  why a message that names a knowledge topic is left alone even when a device is in
  play.

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

#: Words that mean the question is about product knowledge rather than about this
#: device's records. Their presence cancels the correction, so "那它的耗电量是多少"
#: stays a knowledge question instead of becoming a device lookup. The cost is a
#: miss, never a hijack.
#:
#: Warranty and duration words are deliberately *absent*: "还在保修吗", "保修期到多长
#: 时间" and "订单还有多久" are answerable only from the device store, so treating them
#: as knowledge topics would suppress the very follow-up this correction exists for.
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


def mentions_knowledge_topic(question: str) -> bool:
    """True when the message is about product knowledge rather than device records.

    The device label is removed first, so "D2002 的保修期" is judged on "的保修期".
    """

    stripped = DEVICE_ID_PATTERN.sub(" ", question)
    return any(word in stripped for word in KNOWLEDGE_TOPIC_WORDS)


def prefer_device(question: str, device_id: str | None, intent: str) -> bool:
    """Whether a model classification should be corrected to ``device``.

    Each of the four conditions guards against a different mistake:

    * the intent must be replaceable -- never ``ticket``;
    * a device id must already be resolvable -- never invent one;
    * the question must be referential -- a message that mentions a device is not
      automatically about that device's records;
    * the question must not name a knowledge topic -- otherwise a real knowledge
      question gets hijacked into a device listing.
    """

    if intent not in REPLACEABLE_INTENTS:
        return False
    if not device_id:
        return False
    return is_referential(question) and not mentions_knowledge_topic(question)


__all__ = [
    "KNOWLEDGE_TOPIC_WORDS",
    "REFERENTIAL_CUES",
    "REPLACEABLE_INTENTS",
    "is_referential",
    "mentions_knowledge_topic",
    "prefer_device",
]
