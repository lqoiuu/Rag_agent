"""Assemble model providers from validated application settings.

Provider construction lives here so that adapters stay unaware of application
configuration, and settings stay unaware of HTTP details. This is the seam that
later stages call when they need a chat or embedding model.
"""

from __future__ import annotations

import httpx

from rag_agent.config.settings import Settings, get_settings
from rag_agent.providers.qwen import QwenChatModel, QwenEmbeddingModel


def _api_key(settings: Settings) -> str | None:
    secret = settings.qwen_api_key
    return secret.get_secret_value() if secret is not None else None


def build_chat_model(
    settings: Settings | None = None, *, client: httpx.Client | None = None
) -> QwenChatModel:
    """Build the configured chat model.

    Raises :class:`~rag_agent.providers.base.ModelAuthError` when no API key is
    configured, before any network call is made. ``client`` exists so tests and
    advanced callers can supply their own transport.
    """

    resolved = settings if settings is not None else get_settings()
    return QwenChatModel(
        api_key=_api_key(resolved),
        base_url=resolved.qwen_base_url,
        model_name=resolved.qwen_chat_model,
        timeout_seconds=resolved.model_timeout_seconds,
        max_retries=resolved.model_max_retries,
        client=client,
    )


def build_embedding_model(
    settings: Settings | None = None, *, client: httpx.Client | None = None
) -> QwenEmbeddingModel:
    """Build the configured embedding model, with the same key contract."""

    resolved = settings if settings is not None else get_settings()
    return QwenEmbeddingModel(
        api_key=_api_key(resolved),
        base_url=resolved.qwen_base_url,
        model_name=resolved.qwen_embedding_model,
        timeout_seconds=resolved.model_timeout_seconds,
        max_retries=resolved.model_max_retries,
        client=client,
    )
