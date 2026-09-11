"""Intent classification: model proposals plus deterministic fallbacks."""

from __future__ import annotations

import pytest
from agent_test_support import intent_reply

from rag_agent.agent.intent import DEFAULT_MIN_CONFIDENCE, Intent, classify_intent
from rag_agent.providers.fake import FakeChatModel


def test_model_intent_is_used_when_confident() -> None:
    model = FakeChatModel([intent_reply("device", 0.93)])

    decision = classify_intent("我的机器还在保修吗", chat_model=model)

    assert decision.intent is Intent.DEVICE
    assert decision.confidence == 0.93
    assert decision.source == "model"
    assert decision.from_model is True
    assert decision.reason == "scripted"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("knowledge", Intent.KNOWLEDGE),
        ("device", Intent.DEVICE),
        ("ticket", Intent.TICKET),
        ("unknown", Intent.UNKNOWN),
        ("KNOWLEDGE", Intent.KNOWLEDGE),
    ],
)
def test_supported_labels_are_recognised(label: str, expected: Intent) -> None:
    model = FakeChatModel([intent_reply(label)])

    assert classify_intent("问题", chat_model=model).intent is expected


def test_unparseable_reply_falls_back_to_unknown() -> None:
    model = FakeChatModel(["我猜你想问设备信息。"])

    decision = classify_intent("我的机器还在保修吗", chat_model=model)

    assert decision.intent is Intent.UNKNOWN
    assert decision.source == "fallback"
    assert decision.confidence == 0.0
    assert "JSON" in decision.reason


def test_unknown_label_falls_back_to_unknown() -> None:
    model = FakeChatModel([intent_reply("weather")])

    decision = classify_intent("明天天气怎么样", chat_model=model)

    assert decision.intent is Intent.UNKNOWN
    assert decision.source == "fallback"
    assert "weather" in decision.reason


def test_low_confidence_is_downgraded_to_unknown() -> None:
    model = FakeChatModel([intent_reply("device", 0.2)])

    decision = classify_intent("嗯", chat_model=model)

    assert decision.intent is Intent.UNKNOWN
    assert decision.source == "model"
    assert decision.confidence == 0.2
    assert "below" in decision.reason


def test_threshold_is_configurable() -> None:
    model = FakeChatModel([intent_reply("ticket", 0.4)])

    decision = classify_intent("帮我报修", chat_model=model, min_confidence=0.3)

    assert decision.intent is Intent.TICKET


def test_out_of_range_confidence_is_clamped() -> None:
    model = FakeChatModel(['{"intent": "knowledge", "confidence": 5}'])

    decision = classify_intent("充电座放哪里", chat_model=model)

    assert decision.confidence == 1.0
    assert decision.intent is Intent.KNOWLEDGE


def test_non_numeric_confidence_becomes_zero() -> None:
    model = FakeChatModel(['{"intent": "knowledge", "confidence": "high"}'])

    decision = classify_intent("充电座放哪里", chat_model=model)

    assert decision.confidence == 0.0
    assert decision.intent is Intent.UNKNOWN


def test_blank_question_does_not_call_the_model() -> None:
    model = FakeChatModel([intent_reply("knowledge")])

    decision = classify_intent("   ", chat_model=model)

    assert decision.intent is Intent.UNKNOWN
    assert model.call_count == 0


def test_default_threshold_is_documented() -> None:
    assert DEFAULT_MIN_CONFIDENCE == 0.5
