"""The minimal Prompt -> Model -> Parser chain, driven by offline fakes."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_agent.generation import SYSTEM_PROMPT, answer_question, build_minimal_qa_chain
from rag_agent.providers.base import ChatMessage, ChatResponse, ModelTimeoutError
from rag_agent.providers.fake import FakeChatModel


class RecordingModel:
    """Protocol implementation proving the chain depends on behaviour only."""

    def __init__(self) -> None:
        self.seen: list[ChatMessage] = []

    @property
    def model_name(self) -> str:
        return "recording"

    def chat(self, messages: Sequence[ChatMessage], *, temperature: float = 0.0) -> ChatResponse:
        self.seen = list(messages)
        return ChatResponse(text="记录型应答", model="recording", latency_ms=1.5)


def test_chain_renders_system_and_user_messages() -> None:
    model = FakeChatModel(["答案"])

    build_minimal_qa_chain(model).invoke({"question": "充不进电怎么办"})

    messages = model.last_messages
    assert messages is not None
    assert [message.role for message in messages] == ["system", "user"]
    assert messages[0].content == SYSTEM_PROMPT
    assert messages[1].content == "充不进电怎么办"


def test_answer_keeps_the_model_evidence() -> None:
    model = FakeChatModel(["答案"], model_name="fake-chat", latency_ms=12.5)

    answer = answer_question(model, "问题")

    assert answer.text == "答案"
    assert answer.model == "fake-chat"
    assert answer.latency_ms == 12.5
    assert answer.attempts == 1


def test_chain_is_reusable_across_questions() -> None:
    model = FakeChatModel(["第一次", "第二次"])
    chain = build_minimal_qa_chain(model)

    assert chain.invoke({"question": "第一个问题"}).text == "第一次"
    assert chain.invoke({"question": "第二个问题"}).text == "第二次"
    assert model.call_count == 2


def test_chain_accepts_any_protocol_implementation() -> None:
    model = RecordingModel()

    answer = answer_question(model, "你是谁")

    assert answer.text == "记录型应答"
    assert answer.model == "recording"
    assert model.seen[-1].content == "你是谁"


def test_provider_failure_propagates_unchanged() -> None:
    model = FakeChatModel(["不会用到"], errors=[ModelTimeoutError("scripted timeout")])

    with pytest.raises(ModelTimeoutError):
        answer_question(model, "问题")
