"""Deterministic offline providers used by tests and local experiments.

These implementations make the model layer testable without network access and
without an API key, which is what keeps the suite reproducible in CI.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator, Sequence

from rag_agent.providers.base import (
    ChatMessage,
    ChatResponse,
    ChatUsage,
    EmbeddingResponse,
    ModelError,
    ModelRequestError,
)

ChatResponder = Callable[[Sequence[ChatMessage]], str]


class FakeChatModel:
    """Scripted chat model: no network, no credentials, fully predictable.

    Responses are consumed in order; once the queue is exhausted the last
    response repeats so multi-call tests stay readable. Injected ``errors`` are
    raised first, one per call, which is how retry behaviour gets tested.
    """

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        responder: ChatResponder | None = None,
        errors: Sequence[ModelError] = (),
        model_name: str = "fake-chat",
        latency_ms: float = 0.0,
        stream_chunk_chars: int = 4,
    ) -> None:
        if responses is None and responder is None:
            raise ValueError("FakeChatModel needs either responses or a responder")
        if stream_chunk_chars <= 0:
            raise ValueError("stream_chunk_chars must be positive")
        self._responses = list(responses) if responses is not None else []
        self._responder = responder
        self._errors = list(errors)
        self._model_name = model_name
        self._latency_ms = latency_ms
        self._stream_chunk_chars = stream_chunk_chars
        self.calls: list[tuple[ChatMessage, ...]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_messages(self) -> tuple[ChatMessage, ...] | None:
        """Messages of the most recent call, for prompt assertions."""

        return self.calls[-1] if self.calls else None

    def chat(self, messages: Sequence[ChatMessage], *, temperature: float = 0.0) -> ChatResponse:
        """Return the next scripted reply, or raise the next scripted error."""

        recorded = tuple(messages)
        self.calls.append(recorded)

        if self._errors:
            raise self._errors.pop(0)
        if self._responder is not None:
            text = self._responder(recorded)
        elif self._responses:
            text = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        else:
            raise ModelRequestError(
                "fake chat model has no scripted response left", model=self._model_name
            )

        return ChatResponse(
            text=text,
            model=self._model_name,
            latency_ms=self._latency_ms,
            attempts=1,
            usage=ChatUsage(),
        )

    def stream_chat(
        self, messages: Sequence[ChatMessage], *, temperature: float = 0.0
    ) -> Iterator[str]:
        """Yield the scripted reply in small deterministic chunks."""

        text = self.chat(messages, temperature=temperature).text
        size = self._stream_chunk_chars
        for start in range(0, len(text), size):
            yield text[start : start + size]


class FakeEmbeddingModel:
    """Hash-derived embeddings: identical text always maps to one vector.

    Determinism is what makes threshold and ranking assertions in later stages
    reproducible; a random stand-in would produce flaky tests.
    """

    def __init__(
        self,
        *,
        dimension: int = 8,
        model_name: str = "fake-embedding",
        latency_ms: float = 0.0,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self._dimension = dimension
        self._model_name = model_name
        self._latency_ms = latency_ms
        self.calls: list[tuple[str, ...]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        """Return unit-length vectors derived from each text."""

        if not texts:
            raise ModelRequestError("embedding requires at least one text", model=self._model_name)

        batch = tuple(texts)
        self.calls.append(batch)
        return EmbeddingResponse(
            vectors=tuple(self._vector(text) for text in batch),
            model=self._model_name,
            latency_ms=self._latency_ms,
            attempts=1,
        )

    def _vector(self, text: str) -> tuple[float, ...]:
        digest = hashlib.shake_128(text.encode("utf-8")).digest(self._dimension)
        values = [byte / 127.5 - 1.0 for byte in digest]
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            return tuple(0.0 for _ in values)
        return tuple(value / norm for value in values)
