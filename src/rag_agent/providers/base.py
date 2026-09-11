"""Vendor-neutral contracts shared by every model provider.

Chat and embedding capabilities are described as :class:`typing.Protocol`
classes, so application code depends on behaviour instead of a vendor SDK.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One conversation turn in a vendor-neutral form."""

    role: ChatRole
    content: str

    def as_payload(self) -> dict[str, str]:
        """Render this message as an OpenAI-compatible JSON object."""

        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class ChatUsage:
    """Token accounting reported by a provider, when it is available."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """A finished chat call plus the evidence needed for observability."""

    text: str
    model: str
    latency_ms: float
    attempts: int = 1
    usage: ChatUsage = field(default_factory=ChatUsage)


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """One vector per input text, in input order."""

    vectors: tuple[tuple[float, ...], ...]
    model: str
    latency_ms: float
    attempts: int = 1

    @property
    def dimension(self) -> int:
        """Vector width, or 0 for an empty batch."""

        return len(self.vectors[0]) if self.vectors else 0


class ModelError(Exception):
    """Base class for provider failures with an explicit retry contract."""

    code: str = "model_error"
    retryable: bool = False

    def __init__(self, message: str, *, model: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.model = model


class ModelTimeoutError(ModelError):
    """The provider did not answer within the configured timeout."""

    code = "timeout"
    retryable = True


class ModelConnectionError(ModelError):
    """The provider could not be reached at all."""

    code = "connection"
    retryable = True


class ModelRateLimitError(ModelError):
    """The provider rejected the call because of rate limiting."""

    code = "rate_limit"
    retryable = True


class ModelServerError(ModelError):
    """The provider reported an internal failure."""

    code = "server"
    retryable = True


class ModelAuthError(ModelError):
    """Credentials are missing, invalid, or not permitted."""

    code = "auth"


class ModelRequestError(ModelError):
    """The request itself is invalid, so retrying cannot help."""

    code = "request"


class ModelResponseError(ModelError):
    """The provider answered with an unusable or unexpected shape."""

    code = "response"


class ChatModel(Protocol):
    """Chat capability required by the application."""

    @property
    def model_name(self) -> str:
        """Identifier recorded in logs, answers, and evaluation runs."""

        ...

    def chat(self, messages: Sequence[ChatMessage], *, temperature: float = 0.0) -> ChatResponse:
        """Return one assistant reply for the given conversation."""

        ...

    def stream_chat(
        self, messages: Sequence[ChatMessage], *, temperature: float = 0.0
    ) -> Iterator[str]:
        """Yield reply text incrementally, in provider order.

        Deltas are plain text fragments; concatenating them reproduces the full
        reply. Implementations must not retry after the first delta, otherwise
        the caller would see duplicated output.
        """

        ...


class EmbeddingModel(Protocol):
    """Embedding capability required by the application."""

    @property
    def model_name(self) -> str:
        """Identifier recorded in logs and evaluation runs."""

        ...

    def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        """Return one vector per input text, in input order."""

        ...
