"""Error classification and retry behaviour of the Qwen transport."""

from __future__ import annotations

import httpx
import pytest
from model_test_support import (
    API_KEY,
    MESSAGES,
    ScriptedTransport,
    build_chat_model,
    build_client,
    completion_body,
)

from rag_agent.providers.base import (
    ModelAuthError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelRequestError,
    ModelResponseError,
    ModelServerError,
    ModelTimeoutError,
)
from rag_agent.providers.qwen import QwenChatModel

ERROR_BODY = {"error": {"message": "boom"}}


@pytest.mark.parametrize(
    ("status", "expected", "expected_calls"),
    [
        (400, ModelRequestError, 1),
        (401, ModelAuthError, 1),
        (403, ModelAuthError, 1),
        (422, ModelRequestError, 1),
        (429, ModelRateLimitError, 3),
        (500, ModelServerError, 3),
        (503, ModelServerError, 3),
    ],
)
def test_http_status_is_classified(
    status: int, expected: type[ModelError], expected_calls: int
) -> None:
    transport = ScriptedTransport(httpx.Response(status, json=ERROR_BODY))

    with pytest.raises(expected) as excinfo:
        build_chat_model(transport).chat(MESSAGES)

    assert excinfo.value.code == expected.code
    assert excinfo.value.retryable is expected.retryable
    assert transport.call_count == expected_calls


def test_missing_api_key_fails_before_any_request() -> None:
    transport = ScriptedTransport(httpx.Response(200, json=completion_body()))

    with pytest.raises(ModelAuthError):
        QwenChatModel(api_key=None, client=build_client(transport), sleep=lambda _seconds: None)

    assert transport.call_count == 0


def test_blank_api_key_is_rejected() -> None:
    with pytest.raises(ModelAuthError):
        QwenChatModel(api_key="   ")


def test_chat_requires_at_least_one_message() -> None:
    transport = ScriptedTransport(httpx.Response(200, json=completion_body()))

    with pytest.raises(ModelRequestError):
        build_chat_model(transport).chat([])

    assert transport.call_count == 0


def test_timeout_is_retried_up_to_the_limit() -> None:
    transport = ScriptedTransport(httpx.ReadTimeout("timed out"))

    with pytest.raises(ModelTimeoutError):
        build_chat_model(transport, max_retries=2).chat(MESSAGES)

    assert transport.call_count == 3


def test_connection_error_recovers_on_retry() -> None:
    transport = ScriptedTransport(
        httpx.ConnectError("no route to host"),
        httpx.Response(200, json=completion_body("恢复后的回答")),
    )

    response = build_chat_model(transport, max_retries=1).chat(MESSAGES)

    assert response.text == "恢复后的回答"
    assert response.attempts == 2
    assert transport.call_count == 2


def test_rate_limit_then_success_reports_attempt_count() -> None:
    transport = ScriptedTransport(
        httpx.Response(429, json=ERROR_BODY),
        httpx.Response(429, json=ERROR_BODY),
        httpx.Response(200, json=completion_body()),
    )

    response = build_chat_model(transport, max_retries=2).chat(MESSAGES)

    assert response.attempts == 3
    assert transport.call_count == 3


def test_client_error_is_not_retried_even_with_retries_left() -> None:
    transport = ScriptedTransport(httpx.Response(400, json=ERROR_BODY))

    with pytest.raises(ModelRequestError):
        build_chat_model(transport, max_retries=5).chat(MESSAGES)

    assert transport.call_count == 1


def test_non_json_body_is_a_response_error() -> None:
    transport = ScriptedTransport(httpx.Response(200, text="<html>gateway</html>"))

    with pytest.raises(ModelResponseError):
        build_chat_model(transport, max_retries=0).chat(MESSAGES)


def test_json_array_body_is_a_response_error() -> None:
    transport = ScriptedTransport(httpx.Response(200, json=[1, 2, 3]))

    with pytest.raises(ModelResponseError):
        build_chat_model(transport, max_retries=0).chat(MESSAGES)


def test_empty_choices_is_a_response_error() -> None:
    transport = ScriptedTransport(httpx.Response(200, json={"model": "qwen-plus", "choices": []}))

    with pytest.raises(ModelResponseError):
        build_chat_model(transport, max_retries=0).chat(MESSAGES)


def test_blank_content_is_a_response_error() -> None:
    transport = ScriptedTransport(httpx.Response(200, json=completion_body("   ")))

    with pytest.raises(ModelResponseError):
        build_chat_model(transport, max_retries=0).chat(MESSAGES)


def test_schema_drift_is_a_response_error() -> None:
    body = {"choices": [{"message": {"content": 42}}]}
    transport = ScriptedTransport(httpx.Response(200, json=body))

    with pytest.raises(ModelResponseError):
        build_chat_model(transport, max_retries=0).chat(MESSAGES)


def test_response_error_is_not_retried() -> None:
    transport = ScriptedTransport(httpx.Response(200, text="not json"))

    with pytest.raises(ModelResponseError):
        build_chat_model(transport, max_retries=3).chat(MESSAGES)

    assert transport.call_count == 1


def test_retry_contract_is_explicit_per_error_type() -> None:
    assert ModelTimeoutError("x").retryable is True
    assert ModelConnectionError("x").retryable is True
    assert ModelRateLimitError("x").retryable is True
    assert ModelServerError("x").retryable is True
    assert ModelAuthError("x").retryable is False
    assert ModelRequestError("x").retryable is False
    assert ModelResponseError("x").retryable is False

    assert ModelTimeoutError("x").code == "timeout"
    assert ModelServerError("x").code == "server"
    assert ModelAuthError("x").code == "auth"


def test_error_message_does_not_leak_the_api_key() -> None:
    transport = ScriptedTransport(httpx.Response(401, json=ERROR_BODY))

    with pytest.raises(ModelAuthError) as excinfo:
        build_chat_model(transport).chat(MESSAGES)

    assert API_KEY not in str(excinfo.value)
