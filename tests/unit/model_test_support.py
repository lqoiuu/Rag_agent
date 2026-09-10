"""Shared network-free helpers for the provider test modules."""

from __future__ import annotations

import random
from collections.abc import Sequence

import httpx

from rag_agent.providers.base import ChatMessage
from rag_agent.providers.qwen import QwenChatModel, QwenEmbeddingModel

API_KEY = "test-key-not-real"
CHAT_MODEL = "qwen-plus"
EMBEDDING_MODEL = "text-embedding-v4"
MESSAGES: Sequence[ChatMessage] = [ChatMessage(role="user", content="机器充不进电怎么办")]


class ScriptedTransport:
    """Replay scripted HTTP outcomes and record every request made.

    The final outcome repeats once the script is exhausted, which keeps
    "always fails" cases readable.
    """

    def __init__(self, *outcomes: httpx.Response | Exception) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._outcomes) - 1)
        outcome = self._outcomes[index]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def call_count(self) -> int:
        return len(self.requests)


def build_client(transport: ScriptedTransport) -> httpx.Client:
    """Wrap a scripted transport in a client; providers build absolute URLs."""

    return httpx.Client(transport=httpx.MockTransport(transport))


def build_chat_model(
    transport: ScriptedTransport, *, max_retries: int = 2, model_name: str = CHAT_MODEL
) -> QwenChatModel:
    """Build a chat adapter that never sleeps and never touches the network."""

    return QwenChatModel(
        api_key=API_KEY,
        model_name=model_name,
        max_retries=max_retries,
        client=build_client(transport),
        sleep=lambda _seconds: None,
        random_source=random.Random(0),
    )


def build_embedding_model(
    transport: ScriptedTransport, *, max_retries: int = 2, model_name: str = EMBEDDING_MODEL
) -> QwenEmbeddingModel:
    """Build an embedding adapter that never sleeps and never touches the network."""

    return QwenEmbeddingModel(
        api_key=API_KEY,
        model_name=model_name,
        max_retries=max_retries,
        client=build_client(transport),
        sleep=lambda _seconds: None,
        random_source=random.Random(0),
    )


def completion_body(text: str = "请先检查充电座是否通电。") -> dict[str, object]:
    """Build an OpenAI-compatible chat completion payload."""

    return {
        "model": CHAT_MODEL,
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
    }


def embedding_body(vectors: list[list[float]]) -> dict[str, object]:
    """Build an OpenAI-compatible embedding payload."""

    return {"model": EMBEDDING_MODEL, "data": [{"embedding": vector} for vector in vectors]}
