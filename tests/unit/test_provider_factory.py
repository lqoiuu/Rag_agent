"""Settings-driven provider assembly, verified without network access."""

from __future__ import annotations

import json

import httpx
import pytest
from model_test_support import API_KEY, ScriptedTransport, completion_body, embedding_body

from rag_agent.config import build_chat_model, build_embedding_model
from rag_agent.config.settings import Settings
from rag_agent.providers.base import ChatMessage, ModelAuthError, ModelServerError

ERROR_BODY = {"error": {"message": "boom"}}


def make_settings(**overrides: object) -> Settings:
    """Build settings that never read the local .env file."""

    values: dict[str, object] = {"qwen_api_key": API_KEY}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def scripted_model_client(transport: ScriptedTransport) -> httpx.Client:
    """An injected client is a pure transport; the provider owns the URL."""

    return httpx.Client(transport=httpx.MockTransport(transport))


def test_chat_model_uses_configured_model_and_base_url() -> None:
    transport = ScriptedTransport(httpx.Response(200, json=completion_body()))
    settings = make_settings(
        qwen_chat_model="qwen-max",
        qwen_base_url="https://example.invalid/v1",
    )

    model = build_chat_model(settings, client=scripted_model_client(transport))
    response = model.chat([ChatMessage(role="user", content="问题")])

    request = transport.requests[0]
    assert str(request.url) == "https://example.invalid/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert json.loads(request.content)["model"] == "qwen-max"
    assert model.model_name == "qwen-max"
    assert response.text


def test_chat_model_honours_the_configured_retry_limit() -> None:
    transport = ScriptedTransport(httpx.Response(500, json=ERROR_BODY))
    settings = make_settings(model_max_retries=1)

    model = build_chat_model(settings, client=scripted_model_client(transport))

    with pytest.raises(ModelServerError):
        model.chat([ChatMessage(role="user", content="问题")])

    assert transport.call_count == 2


def test_embedding_model_uses_the_configured_embedding_model() -> None:
    transport = ScriptedTransport(httpx.Response(200, json=embedding_body([[0.1, 0.2]])))
    settings = make_settings(qwen_embedding_model="text-embedding-v3")

    model = build_embedding_model(settings, client=scripted_model_client(transport))
    response = model.embed(["主刷卡住"])

    assert json.loads(transport.requests[0].content)["model"] == "text-embedding-v3"
    assert response.dimension == 2


def test_missing_api_key_prevents_building_a_model() -> None:
    settings = make_settings(qwen_api_key=None)

    with pytest.raises(ModelAuthError):
        build_chat_model(settings)

    with pytest.raises(ModelAuthError):
        build_embedding_model(settings)
