"""Request and response contract of the Qwen adapters, verified without network."""

from __future__ import annotations

import json

import httpx
import pytest
from model_test_support import (
    API_KEY,
    CHAT_MODEL,
    EMBEDDING_MODEL,
    ScriptedTransport,
    build_chat_model,
    build_client,
    build_embedding_model,
    completion_body,
    embedding_body,
)

from rag_agent.providers.base import ChatMessage, ModelRequestError, ModelResponseError
from rag_agent.providers.qwen import DEFAULT_BASE_URL, QwenChatModel


def test_chat_request_shape_and_response_metadata() -> None:
    transport = ScriptedTransport(httpx.Response(200, json=completion_body("请检查充电座。")))
    model = build_chat_model(transport)

    response = model.chat(
        [ChatMessage(role="system", content="规则"), ChatMessage(role="user", content="问题")],
        temperature=0.3,
    )

    request = transport.requests[0]
    assert request.method == "POST"
    assert request.url.path.endswith("/chat/completions")
    assert request.headers["authorization"] == f"Bearer {API_KEY}"

    body = json.loads(request.content)
    assert body["model"] == CHAT_MODEL
    assert body["stream"] is False
    assert body["temperature"] == 0.3
    assert body["messages"] == [
        {"role": "system", "content": "规则"},
        {"role": "user", "content": "问题"},
    ]

    assert response.text == "请检查充电座。"
    assert response.model == CHAT_MODEL
    assert response.attempts == 1
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 8
    assert response.latency_ms >= 0.0


def test_default_base_url_targets_the_dashscope_compatible_endpoint() -> None:
    transport = ScriptedTransport(httpx.Response(200, json=completion_body()))

    build_chat_model(transport).chat([ChatMessage(role="user", content="问题")])

    assert DEFAULT_BASE_URL == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert str(transport.requests[0].url).startswith(DEFAULT_BASE_URL)


def test_trimmed_content_is_returned_without_surrounding_space() -> None:
    transport = ScriptedTransport(httpx.Response(200, json=completion_body("  答案  ")))

    response = build_chat_model(transport).chat([ChatMessage(role="user", content="问题")])

    assert response.text == "答案"


def test_embedding_request_shape_and_vectors() -> None:
    transport = ScriptedTransport(
        httpx.Response(200, json=embedding_body([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    )
    model = build_embedding_model(transport)

    response = model.embed(["第一段", "第二段"])

    body = json.loads(transport.requests[0].content)
    assert body == {"model": EMBEDDING_MODEL, "input": ["第一段", "第二段"]}
    assert response.vectors == ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    assert response.dimension == 3
    assert response.model == EMBEDDING_MODEL
    assert response.attempts == 1


def test_embedding_rejects_blank_text_without_calling_the_service() -> None:
    transport = ScriptedTransport(httpx.Response(200, json=embedding_body([[0.1]])))

    with pytest.raises(ModelRequestError):
        build_embedding_model(transport).embed(["主刷卡住", "   "])

    assert transport.call_count == 0


def test_embedding_requires_at_least_one_text() -> None:
    transport = ScriptedTransport(httpx.Response(200, json=embedding_body([[0.1]])))

    with pytest.raises(ModelRequestError):
        build_embedding_model(transport).embed([])

    assert transport.call_count == 0


def test_embedding_count_mismatch_is_rejected() -> None:
    transport = ScriptedTransport(httpx.Response(200, json=embedding_body([[0.1]])))

    with pytest.raises(ModelResponseError):
        build_embedding_model(transport).embed(["一", "二"])


def test_embedding_dimension_mismatch_is_rejected() -> None:
    transport = ScriptedTransport(httpx.Response(200, json=embedding_body([[0.1, 0.2], [0.3]])))

    with pytest.raises(ModelResponseError):
        build_embedding_model(transport).embed(["一", "二"])


def test_injected_client_stays_open_because_the_caller_owns_it() -> None:
    transport = ScriptedTransport(httpx.Response(200, json=completion_body()))
    client = build_client(transport)
    model = QwenChatModel(api_key=API_KEY, client=client, sleep=lambda _seconds: None)

    model.close()

    assert client.is_closed is False
    client.close()


def test_owned_client_close_is_idempotent() -> None:
    model = QwenChatModel(api_key=API_KEY)

    model.close()
    model.close()


def test_context_manager_returns_a_usable_model() -> None:
    transport = ScriptedTransport(httpx.Response(200, json=completion_body("答案")))

    with build_chat_model(transport) as model:
        response = model.chat([ChatMessage(role="user", content="问题")])

    assert response.text == "答案"
