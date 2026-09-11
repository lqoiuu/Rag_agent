"""The deterministic routing correction: hit cases and, more importantly, misses.

The correction exists because the model classified the same question differently
depending on whether history was in the prompt, and because it classified a plainly
named device as a knowledge question. Its value depends entirely on how narrow it is, so
most of these tests are about what it must *refuse* to do.
"""

from __future__ import annotations

import pytest

from rag_agent.agent.routing import (
    DEVICE_RECORD_CUES,
    KNOWLEDGE_TOPIC_WORDS,
    asks_about_device_records,
    is_referential,
    mentions_knowledge_topic,
    prefer_device,
)

FOLLOW_UP = "那它的保修期是多久"

#: The exact question that was refused in the running UI, screenshot and all: the device
#: is named outright, the model said "knowledge", and the user got a refusal about the
#: manual not mentioning D2002.
REGRESSION = "D2002 还在保修吗"


@pytest.mark.parametrize(
    ("question", "device_id", "intent", "expected"),
    [
        # 指代式追问 + 上下文里已有设备号 + 模型没判成 device
        (FOLLOW_UP, "D2002", "knowledge", True),
        (FOLLOW_UP, "D2002", "unknown", True),
        ("这个还在保修吗", "D2002", "knowledge", True),
        ("上述设备的订单呢", "D2002", "unknown", True),
        # 直接点名设备、问的是只有设备存储才有的字段：没有指代词也必须改判
        (REGRESSION, "D2002", "knowledge", True),
        ("D2002 的订单呢", "D2002", "knowledge", True),
        ("D2002 什么时候过保", "D2002", "unknown", True),
        ("我的设备保修到什么时候", "D2002", "knowledge", True),
    ],
)
def test_corrects_a_device_record_question(
    question: str, device_id: str | None, intent: str, expected: bool
) -> None:
    assert prefer_device(question, device_id, intent) is expected


@pytest.mark.parametrize(
    ("question", "device_id", "intent"),
    [
        # 模型已经判对，无须改判
        (REGRESSION, "D2002", "device"),
        # 报修是显式动作，绝不能被改写成设备查询
        ("帮我把 D2001 报修", "D2001", "ticket"),
        ("那它就报修吧", "D2001", "ticket"),
        # 既无指代、也不问设备记录字段：只是提到设备而已，不能改判
        ("D2002 是哪个厂家生产的", "D2002", "knowledge"),
        # 上下文里根本没有设备号，不能凭空造一个
        (REGRESSION, None, "knowledge"),
        (FOLLOW_UP, None, "knowledge"),
        (FOLLOW_UP, "", "knowledge"),
        # 指代或记录词 + 知识话题词：宁可漏改，也不要把知识问题劫持成设备列表
        ("那它的耗电量是多少", "D2002", "knowledge"),
        ("那这个怎么拆滤芯", "D2002", "knowledge"),
        ("这台设备的额定电压呢", "D2002", "knowledge"),
        ("D2002 的保修期内该怎么清洁滤网", "D2002", "knowledge"),
    ],
)
def test_leaves_everything_else_alone(question: str, device_id: str | None, intent: str) -> None:
    assert prefer_device(question, device_id, intent) is False


def test_referential_detection_needs_a_cue() -> None:
    assert is_referential(FOLLOW_UP) is True
    assert is_referential("它的保修期呢") is True
    assert is_referential(REGRESSION) is False


def test_device_record_detection_is_about_fields_not_pronouns() -> None:
    """A named device with a record question must be caught without any pronoun."""

    assert asks_about_device_records(REGRESSION) is True
    assert asks_about_device_records("D2002 的订单呢") is True
    assert asks_about_device_records("D2002 是哪个厂家生产的") is False


def test_knowledge_topic_detection_ignores_the_device_label() -> None:
    """The device label is stripped first, so only the topic decides."""

    assert mentions_knowledge_topic("那它的耗电量是多少") is True
    assert mentions_knowledge_topic("那它的订单呢") is False
    assert mentions_knowledge_topic("D2001") is False
    # 保修与时长只存在于设备记录里，不能算知识话题，否则会掐掉要修的那种追问
    assert mentions_knowledge_topic("那它的保修期是多久") is False
    assert mentions_knowledge_topic(REGRESSION) is False


def test_the_two_word_lists_never_contradict_each_other() -> None:
    """A word may not be both a device-record field and a knowledge topic.

    This guard exists because the contradiction happened twice while writing the rule:
    ``保修`` and then the duration words were listed as knowledge topics, which made the
    correction a no-op for exactly the questions it was written for. The behaviour tests
    above caught each instance; this one makes the next instance impossible to miss.
    """

    overlap = set(DEVICE_RECORD_CUES) & set(KNOWLEDGE_TOPIC_WORDS)
    assert overlap == set(), f"这些词同时出现在两张表里：{sorted(overlap)}"
