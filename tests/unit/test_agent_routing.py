"""The deterministic routing correction: hit cases and, more importantly, misses.

The correction exists because the model classified the same follow-up differently
depending on whether history was in the prompt. Its value depends entirely on how
narrow it is, so most of these tests are about what it must *refuse* to do.
"""

from __future__ import annotations

import pytest

from rag_agent.agent.routing import (
    is_referential,
    mentions_knowledge_topic,
    prefer_device,
)

FOLLOW_UP = "那它的保修期是多久"


@pytest.mark.parametrize(
    ("question", "device_id", "intent", "expected"),
    [
        # 命中：指代式追问 + 上下文里已有设备号 + 模型没判成 device
        (FOLLOW_UP, "D2002", "knowledge", True),
        (FOLLOW_UP, "D2002", "unknown", True),
        ("这个还在保修吗", "D2002", "knowledge", True),
        ("上述设备的订单呢", "D2002", "unknown", True),
    ],
)
def test_corrects_a_referential_follow_up(
    question: str, device_id: str, intent: str, expected: bool
) -> None:
    assert prefer_device(question, device_id, intent) is expected


@pytest.mark.parametrize(
    ("question", "device_id", "intent"),
    [
        # 模型已经判对，无须改判
        ("D2002 还在保修吗", "D2002", "device"),
        # 报修是显式动作，绝不能被改写成设备查询
        ("帮我把 D2001 报修", "D2001", "ticket"),
        ("那它就报修吧", "D2001", "ticket"),
        # 没有指代线索：消息提到设备不等于在问这台设备的记录
        ("D2002 的订单", "D2002", "knowledge"),
        # 上下文里根本没有设备号，不能凭空造一个
        (FOLLOW_UP, None, "knowledge"),
        (FOLLOW_UP, "", "knowledge"),
        # 指代 + 知识话题词：宁可漏改，也不要把知识问题劫持成设备列表
        ("那它的耗电量是多少", "D2002", "knowledge"),
        ("那这个怎么拆滤芯", "D2002", "knowledge"),
        ("这台设备的额定电压呢", "D2002", "knowledge"),
    ],
)
def test_leaves_everything_else_alone(question: str, device_id: str | None, intent: str) -> None:
    assert prefer_device(question, device_id, intent) is False


def test_referential_detection_needs_a_cue() -> None:
    assert is_referential(FOLLOW_UP) is True
    assert is_referential("它的保修期呢") is True
    assert is_referential("D2002 的保修期是多久") is False


def test_knowledge_topic_detection_ignores_the_device_label() -> None:
    """The device label is stripped first, so only the topic decides."""

    assert mentions_knowledge_topic("那它的耗电量是多少") is True
    assert mentions_knowledge_topic("那它的订单呢") is False
    assert mentions_knowledge_topic("D2001") is False
    # 保修与时长只存在于设备记录里，不能算知识话题，否则会掐掉要修的那种追问
    assert mentions_knowledge_topic("那它的保修期是多久") is False
