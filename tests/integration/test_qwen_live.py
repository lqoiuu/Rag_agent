"""Live checks that need a real DashScope key and network access.

Skipped by default. To run them, set ``RAG_AGENT_RUN_LIVE_TESTS=1`` and provide
``RAG_AGENT_QWEN_API_KEY`` either in the shell environment or in the local
``.env`` file. Providers are built through the settings-driven factory, so these
tests also prove the configuration wiring. The key is never printed or written.
"""

from __future__ import annotations

import os

import pytest

from rag_agent.config import build_chat_model, build_embedding_model
from rag_agent.config.settings import Settings
from rag_agent.providers.base import ChatMessage

RUN_LIVE_TESTS_ENV = "RAG_AGENT_RUN_LIVE_TESTS"


def _live_api_key() -> str | None:
    if os.environ.get(RUN_LIVE_TESTS_ENV) != "1":
        return None
    secret = Settings().qwen_api_key
    return secret.get_secret_value() if secret is not None else None


LIVE_API_KEY = _live_api_key()

pytestmark = pytest.mark.skipif(
    LIVE_API_KEY is None,
    reason=f"set {RUN_LIVE_TESTS_ENV}=1 and RAG_AGENT_QWEN_API_KEY to run live checks",
)


def test_live_chat_answers_a_fixed_question() -> None:
    assert LIVE_API_KEY is not None
    question = "用一句话说明扫地机器人充电座应该放在什么位置"

    with build_chat_model() as model:
        response = model.chat([ChatMessage(role="user", content=question)])

    assert response.text.strip()
    assert response.attempts >= 1
    assert response.latency_ms > 0.0


def test_live_embedding_returns_consistent_dimensions() -> None:
    assert LIVE_API_KEY is not None

    with build_embedding_model() as model:
        response = model.embed(["主刷卡住", "充电失败"])

    assert len(response.vectors) == 2
    assert response.dimension > 0
    assert len(response.vectors[0]) == len(response.vectors[1])
