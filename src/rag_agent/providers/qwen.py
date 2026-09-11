"""Tongyi Qianwen provider built on the DashScope OpenAI-compatible HTTP API.

The module depends on the provider protocols in :mod:`rag_agent.providers.base`,
not the other way around: replacing DashScope means adding a sibling module, not
editing business code.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final

import httpx
from pydantic import BaseModel, ValidationError

from rag_agent.providers.base import (
    ChatMessage,
    ChatResponse,
    ChatUsage,
    EmbeddingResponse,
    ModelAuthError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelRequestError,
    ModelResponseError,
    ModelServerError,
    ModelTimeoutError,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL: Final[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_CHAT_MODEL: Final[str] = "qwen-plus"
DEFAULT_EMBEDDING_MODEL: Final[str] = "text-embedding-v4"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_MAX_RETRIES: Final[int] = 2
DEFAULT_BACKOFF_SECONDS: Final[float] = 0.5
_ERROR_DETAIL_LIMIT: Final[int] = 200


class _UsagePayload(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class _ChatMessagePayload(BaseModel):
    content: str | None = None


class _ChatChoicePayload(BaseModel):
    message: _ChatMessagePayload


class _ChatCompletionPayload(BaseModel):
    model: str | None = None
    choices: list[_ChatChoicePayload]
    usage: _UsagePayload | None = None


class _EmbeddingItemPayload(BaseModel):
    embedding: list[float]


class _EmbeddingPayload(BaseModel):
    model: str | None = None
    data: list[_EmbeddingItemPayload]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded exponential backoff with jitter."""

    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS

    def delay_seconds(self, attempt: int, rng: random.Random) -> float:
        """Return the delay before the next attempt; ``attempt`` is zero-based."""

        ceiling = self.backoff_seconds * (2.0**attempt)
        return ceiling * (0.5 + rng.random() / 2.0)


@dataclass(frozen=True, slots=True)
class _HttpResult:
    """A successful HTTP round trip including retry evidence."""

    payload: dict[str, Any]
    attempts: int
    latency_ms: float


def _parse_payload[T: BaseModel](
    payload: dict[str, Any], payload_type: type[T], *, model_name: str
) -> T:
    """Validate a raw JSON body, turning schema drift into a classified error."""

    try:
        return payload_type.model_validate(payload)
    except ValidationError as exc:
        raise ModelResponseError(
            f"provider returned an unexpected response shape ({exc.error_count()} errors)",
            model=model_name,
        ) from exc


def _response_detail(response: httpx.Response) -> str:
    """Return a short, single-line excerpt of an error body."""

    return response.text.strip().replace("\n", " ")[:_ERROR_DETAIL_LIMIT]


def _parse_stream_line(line: str) -> str | None:
    """Extract one text delta from a server-sent-event line."""

    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    data = stripped[len("data:") :].strip()
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return None
    content = delta.get("content")
    return content if isinstance(content, str) and content else None


def _require_api_key(api_key: str | None) -> str:
    """Fail before any network call when credentials are missing."""

    if api_key is None or not api_key.strip():
        raise ModelAuthError(
            "no API key configured; set RAG_AGENT_QWEN_API_KEY before calling the model service"
        )
    return api_key.strip()


class _QwenTransport:
    """Shared HTTP plumbing for the chat and embedding endpoints."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout_seconds: float,
        retry_policy: RetryPolicy,
        client: httpx.Client | None,
        sleep: Callable[[float], None],
        random_source: random.Random | None,
        clock: Callable[[], float],
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._retry_policy = retry_policy
        self._client = client
        self._owns_client = client is None
        self._sleep = sleep
        self._random = random_source if random_source is not None else random.Random()
        self._clock = clock

    def _http_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout_seconds)
        return self._client

    def close(self) -> None:
        """Close the HTTP client, but only when this provider created it."""

        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _error_for_status(self, response: httpx.Response) -> ModelError:
        status = response.status_code
        message = f"model service returned HTTP {status}"
        detail = _response_detail(response)
        if detail:
            message = f"{message}: {detail}"
        if status in (401, 403):
            return ModelAuthError(message, model=self._model_name)
        if status == 429:
            return ModelRateLimitError(message, model=self._model_name)
        if status >= 500:
            return ModelServerError(message, model=self._model_name)
        return ModelRequestError(message, model=self._model_name)

    def _attempt(
        self, client: httpx.Client, path: str, payload: dict[str, Any]
    ) -> dict[str, Any] | ModelError:
        """Perform one HTTP attempt and return either JSON data or an error."""

        try:
            response = client.post(f"{self._base_url}{path}", json=payload, headers=self._headers())
        except httpx.TimeoutException as exc:
            return ModelTimeoutError(
                f"model request exceeded {self._timeout_seconds}s: {exc}",
                model=self._model_name,
            )
        except httpx.HTTPError as exc:
            return ModelConnectionError(
                f"model request could not be completed: {exc}", model=self._model_name
            )

        if response.status_code != httpx.codes.OK:
            return self._error_for_status(response)

        try:
            body = response.json()
        except ValueError:
            return ModelResponseError(
                "model service returned a body that is not JSON", model=self._model_name
            )
        if not isinstance(body, dict):
            return ModelResponseError(
                "model service returned JSON that is not an object", model=self._model_name
            )
        return body

    def post_json(self, path: str, payload: dict[str, Any]) -> _HttpResult:
        """POST a JSON body, retrying only failures that can succeed later."""

        client = self._http_client()
        started = self._clock()
        attempts = 0
        last_error: ModelError | None = None

        while attempts <= self._retry_policy.max_retries:
            attempts += 1
            outcome = self._attempt(client, path, payload)
            if not isinstance(outcome, ModelError):
                latency_ms = (self._clock() - started) * 1000.0
                return _HttpResult(payload=outcome, attempts=attempts, latency_ms=latency_ms)

            last_error = outcome
            if not outcome.retryable or attempts > self._retry_policy.max_retries:
                break
            delay = self._retry_policy.delay_seconds(attempts - 1, self._random)
            LOGGER.warning(
                "qwen request failed code=%s attempt=%d/%d retry_in_ms=%.1f",
                outcome.code,
                attempts,
                self._retry_policy.max_retries,
                delay * 1000.0,
            )
            self._sleep(delay)

        if last_error is None:
            raise ModelConnectionError("model request was never attempted", model=self._model_name)
        raise last_error

    def stream_json(self, path: str, payload: dict[str, Any]) -> Iterator[str]:
        """Stream a server-sent-event reply, yielding text deltas.

        Retries only happen before the first delta; once output has been handed
        to the caller, a retry would duplicate text.
        """

        client = self._http_client()
        attempts = 0
        while True:
            attempts += 1
            yielded = False
            try:
                try:
                    with client.stream(
                        "POST",
                        f"{self._base_url}{path}",
                        json=payload,
                        headers=self._headers(),
                    ) as response:
                        if response.status_code != httpx.codes.OK:
                            response.read()
                            raise self._error_for_status(response)
                        for line in response.iter_lines():
                            delta = _parse_stream_line(line)
                            if delta is None:
                                continue
                            yielded = True
                            yield delta
                except httpx.TimeoutException as exc:
                    raise ModelTimeoutError(
                        f"model stream stalled: {exc}", model=self._model_name
                    ) from exc
                except httpx.HTTPError as exc:
                    raise ModelConnectionError(
                        f"model stream could not be completed: {exc}", model=self._model_name
                    ) from exc
                return
            except ModelError as exc:
                if yielded or not exc.retryable or attempts > self._retry_policy.max_retries:
                    raise
                delay = self._retry_policy.delay_seconds(attempts - 1, self._random)
                LOGGER.warning(
                    "qwen stream failed code=%s attempt=%d/%d retry_in_ms=%.1f",
                    exc.code,
                    attempts,
                    self._retry_policy.max_retries,
                    delay * 1000.0,
                )
                self._sleep(delay)


class QwenChatModel:
    """Chat completion adapter for the DashScope compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model_name: str = DEFAULT_CHAT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_source: random.Random | None = None,
    ) -> None:
        self._model_name = model_name
        self._transport = _QwenTransport(
            api_key=_require_api_key(api_key),
            base_url=base_url,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            retry_policy=RetryPolicy(max_retries=max_retries, backoff_seconds=backoff_seconds),
            client=client,
            sleep=sleep,
            random_source=random_source,
            clock=time.perf_counter,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def close(self) -> None:
        """Release the HTTP client owned by this provider."""

        self._transport.close()

    def __enter__(self) -> QwenChatModel:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def chat(self, messages: Sequence[ChatMessage], *, temperature: float = 0.0) -> ChatResponse:
        """Return one assistant reply, retrying only retryable failures."""

        if not messages:
            raise ModelRequestError("chat requires at least one message", model=self._model_name)

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": [message.as_payload() for message in messages],
            "temperature": temperature,
            "stream": False,
        }
        result = self._transport.post_json("/chat/completions", payload)
        completion = _parse_payload(
            result.payload, _ChatCompletionPayload, model_name=self._model_name
        )
        if not completion.choices:
            raise ModelResponseError("chat response contained no choices", model=self._model_name)

        text = (completion.choices[0].message.content or "").strip()
        if not text:
            raise ModelResponseError(
                "chat response contained empty content", model=self._model_name
            )

        usage = ChatUsage(
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else None,
            completion_tokens=completion.usage.completion_tokens if completion.usage else None,
        )
        resolved_model = completion.model or self._model_name
        LOGGER.info(
            "qwen chat ok model=%s attempts=%d prompt_tokens=%s completion_tokens=%s",
            resolved_model,
            result.attempts,
            usage.prompt_tokens,
            usage.completion_tokens,
        )
        return ChatResponse(
            text=text,
            model=resolved_model,
            latency_ms=result.latency_ms,
            attempts=result.attempts,
            usage=usage,
        )

    def stream_chat(
        self, messages: Sequence[ChatMessage], *, temperature: float = 0.0
    ) -> Iterator[str]:
        """Yield the reply as the provider sends it, delta by delta."""

        if not messages:
            raise ModelRequestError("chat requires at least one message", model=self._model_name)

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": [message.as_payload() for message in messages],
            "temperature": temperature,
            "stream": True,
        }
        yield from self._transport.stream_json("/chat/completions", payload)


class QwenEmbeddingModel:
    """Embedding adapter for the DashScope compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_source: random.Random | None = None,
    ) -> None:
        self._model_name = model_name
        self._transport = _QwenTransport(
            api_key=_require_api_key(api_key),
            base_url=base_url,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            retry_policy=RetryPolicy(max_retries=max_retries, backoff_seconds=backoff_seconds),
            client=client,
            sleep=sleep,
            random_source=random_source,
            clock=time.perf_counter,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def close(self) -> None:
        """Release the HTTP client owned by this provider."""

        self._transport.close()

    def __enter__(self) -> QwenEmbeddingModel:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        """Return one vector per input text, validating batch consistency."""

        if not texts:
            raise ModelRequestError("embedding requires at least one text", model=self._model_name)
        if any(not text.strip() for text in texts):
            raise ModelRequestError("embedding inputs must not be blank", model=self._model_name)

        payload: dict[str, Any] = {"model": self._model_name, "input": list(texts)}
        result = self._transport.post_json("/embeddings", payload)
        parsed = _parse_payload(result.payload, _EmbeddingPayload, model_name=self._model_name)
        if len(parsed.data) != len(texts):
            raise ModelResponseError(
                f"expected {len(texts)} vectors but received {len(parsed.data)}",
                model=self._model_name,
            )

        vectors = tuple(tuple(float(value) for value in item.embedding) for item in parsed.data)
        if len({len(vector) for vector in vectors}) > 1:
            raise ModelResponseError(
                "embedding vectors have inconsistent dimensions", model=self._model_name
            )
        return EmbeddingResponse(
            vectors=vectors,
            model=parsed.model or self._model_name,
            latency_ms=result.latency_ms,
            attempts=result.attempts,
        )
