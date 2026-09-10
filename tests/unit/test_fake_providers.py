"""Behaviour of the offline fake providers that make the suite reproducible."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from rag_agent.providers.base import ChatMessage, ModelRequestError, ModelTimeoutError
from rag_agent.providers.fake import FakeChatModel, FakeEmbeddingModel


def test_fake_chat_consumes_scripted_responses_in_order() -> None:
    model = FakeChatModel(["第一答", "第二答"])
    messages = [ChatMessage(role="user", content="问题")]

    assert model.chat(messages).text == "第一答"
    assert model.chat(messages).text == "第二答"
    assert model.chat(messages).text == "第二答"
    assert model.call_count == 3


def test_fake_chat_records_messages_for_prompt_assertions() -> None:
    model = FakeChatModel(["ok"])
    sent = (ChatMessage(role="system", content="规则"), ChatMessage(role="user", content="你好"))

    model.chat(sent)

    assert model.last_messages == sent


def test_fake_chat_raises_injected_errors_before_answering() -> None:
    model = FakeChatModel(["成功"], errors=[ModelTimeoutError("scripted timeout")])
    messages = [ChatMessage(role="user", content="问题")]

    with pytest.raises(ModelTimeoutError):
        model.chat(messages)

    assert model.chat(messages).text == "成功"


def test_fake_chat_without_script_is_rejected() -> None:
    with pytest.raises(ValueError, match="responses or a responder"):
        FakeChatModel()


def test_fake_chat_responder_receives_the_messages() -> None:
    seen: list[tuple[ChatMessage, ...]] = []

    def responder(messages: Sequence[ChatMessage]) -> str:
        seen.append(tuple(messages))
        return "由 responder 生成"

    model = FakeChatModel(responder=responder)

    assert model.chat([ChatMessage(role="user", content="q")]).text == "由 responder 生成"
    assert seen[0][0].content == "q"


def test_fake_chat_reports_the_configured_model_and_latency() -> None:
    model = FakeChatModel(["ok"], model_name="fake-chat", latency_ms=7.5)

    response = model.chat([ChatMessage(role="user", content="q")])

    assert response.model == "fake-chat"
    assert response.latency_ms == 7.5
    assert response.attempts == 1


def test_fake_embedding_is_deterministic_and_unit_length() -> None:
    model = FakeEmbeddingModel(dimension=8)

    first = model.embed(["主刷卡住"]).vectors[0]
    second = model.embed(["主刷卡住"]).vectors[0]
    other = model.embed(["充电故障"]).vectors[0]

    assert first == second
    assert first != other
    assert len(first) == 8
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0, rel_tol=1e-9)


def test_fake_embedding_keeps_batch_order_and_records_calls() -> None:
    model = FakeEmbeddingModel()

    response = model.embed(["第一段", "第二段"])

    expected_first = FakeEmbeddingModel().embed(["第一段"]).vectors[0]
    assert response.vectors[0] == expected_first
    assert len(response.vectors) == 2
    assert model.call_count == 1
    assert model.calls[0] == ("第一段", "第二段")


def test_fake_embedding_rejects_empty_batch() -> None:
    model = FakeEmbeddingModel()

    with pytest.raises(ModelRequestError):
        model.embed([])


def test_fake_embedding_rejects_non_positive_dimension() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        FakeEmbeddingModel(dimension=0)
