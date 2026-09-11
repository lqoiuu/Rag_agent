"""Token streaming: SSE parsing, retry rules and the fake equivalent."""

from __future__ import annotations

import json

import httpx
import pytest
from model_test_support import ScriptedTransport, build_chat_model

from rag_agent.providers.base import (
    ChatMessage,
    ModelAuthError,
    ModelConnectionError,
    ModelRequestError,
    ModelTimeoutError,
)
from rag_agent.providers.fake import FakeChatModel

MESSAGES = [ChatMessage(role="user", content="机器充不进电怎么办")]


def sse_body(*contents: str, done: bool = True) -> str:
    lines = [
        f"data: {json.dumps({'choices': [{'delta': {'content': content}}]}, ensure_ascii=False)}"
        for content in contents
    ]
    if done:
        lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def stream_response(text: str) -> httpx.Response:
    return httpx.Response(200, text=text)


def test_stream_yields_deltas_in_order() -> None:
    transport = ScriptedTransport(stream_response(sse_body("主刷", "卡住", "时先断电。")))
    model = build_chat_model(transport)

    deltas = list(model.stream_chat(MESSAGES))

    assert deltas == ["主刷", "卡住", "时先断电。"]
    assert "".join(deltas) == "主刷卡住时先断电。"


def test_stream_ignores_noise_and_the_done_marker() -> None:
    body = (
        ": keep-alive\n\n"
        "event: message\n\n"
        'data: {"choices":[{"delta":{"content":"答案"}}]}\n\n'
        "data: not json\n\n"
        'data: {"choices":[]}\n\n'
        'data: {"choices":[{"delta":{}}]}\n\n'
        "data: [DONE]\n\n"
    )
    model = build_chat_model(ScriptedTransport(stream_response(body)))

    assert list(model.stream_chat(MESSAGES)) == ["答案"]


def test_stream_without_content_yields_nothing() -> None:
    model = build_chat_model(ScriptedTransport(stream_response("data: [DONE]\n\n")))

    assert list(model.stream_chat(MESSAGES)) == []


def test_stream_reports_http_errors_before_any_delta() -> None:
    transport = ScriptedTransport(httpx.Response(401, json={"error": {"message": "nope"}}))
    model = build_chat_model(transport)

    with pytest.raises(ModelAuthError):
        list(model.stream_chat(MESSAGES))

    assert transport.call_count == 1


def test_stream_retries_a_retryable_failure_before_any_delta() -> None:
    transport = ScriptedTransport(
        httpx.ConnectError("no route to host"),
        stream_response(sse_body("恢复后的文本")),
    )
    model = build_chat_model(transport, max_retries=1)

    deltas = list(model.stream_chat(MESSAGES))

    assert deltas == ["恢复后的文本"]
    assert transport.call_count == 2


def test_stream_does_not_retry_after_the_first_delta() -> None:
    def exploding() -> object:
        yield b'data: {"choices":[{"delta":{"content":"\xe5\xbc\x80\xe5\xa7\x8b"}}]}\n\n'
        raise httpx.ReadError("connection dropped")

    transport = ScriptedTransport(httpx.Response(200, content=exploding()))
    model = build_chat_model(transport, max_retries=3)

    collected: list[str] = []
    with pytest.raises(ModelConnectionError):
        for delta in model.stream_chat(MESSAGES):
            collected.append(delta)

    assert collected == ["开始"]
    assert transport.call_count == 1


def test_stream_requires_messages() -> None:
    model = build_chat_model(ScriptedTransport(stream_response(sse_body("x"))))

    with pytest.raises(ModelRequestError):
        list(model.stream_chat([]))


def test_fake_model_streams_its_scripted_reply_in_chunks() -> None:
    model = FakeChatModel(["完整的一段回答"], stream_chunk_chars=3)

    deltas = list(model.stream_chat(MESSAGES))

    assert len(deltas) > 1
    assert all(len(delta) <= 3 for delta in deltas)
    assert "".join(deltas) == "完整的一段回答"
    assert model.call_count == 1


def test_fake_model_stream_raises_injected_errors() -> None:
    model = FakeChatModel(["未使用"], errors=[ModelTimeoutError("scripted")])

    with pytest.raises(ModelTimeoutError):
        list(model.stream_chat(MESSAGES))
