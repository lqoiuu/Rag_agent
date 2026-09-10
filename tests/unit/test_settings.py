from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_agent.config.settings import PROJECT_ROOT, Settings
from rag_agent.providers.qwen import (
    DEFAULT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
)

MODEL_ENV_VARS = (
    "RAG_AGENT_QWEN_BASE_URL",
    "RAG_AGENT_QWEN_CHAT_MODEL",
    "RAG_AGENT_QWEN_EMBEDDING_MODEL",
    "RAG_AGENT_MODEL_TIMEOUT_SECONDS",
    "RAG_AGENT_MODEL_MAX_RETRIES",
)


def test_relative_paths_resolve_from_project_root() -> None:
    settings = Settings(
        _env_file=None,
        data_dir=Path("data/test"),
        chroma_dir=Path("data/test/chroma"),
        sqlite_path=Path("data/test/rag_agent.sqlite3"),
    )

    assert settings.data_dir == (PROJECT_ROOT / "data/test").resolve()
    assert settings.chroma_dir == (PROJECT_ROOT / "data/test/chroma").resolve()
    assert settings.sqlite_path == (PROJECT_ROOT / "data/test/rag_agent.sqlite3").resolve()


def test_api_key_is_optional_during_stage_one() -> None:
    settings = Settings(_env_file=None, qwen_api_key=None)

    assert settings.qwen_api_key is None


def test_model_settings_default_to_provider_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in MODEL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.qwen_base_url == DEFAULT_BASE_URL
    assert settings.qwen_chat_model == DEFAULT_CHAT_MODEL
    assert settings.qwen_embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.model_timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert settings.model_max_retries == DEFAULT_MAX_RETRIES


def test_model_settings_can_be_overridden_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_AGENT_QWEN_CHAT_MODEL", "qwen-max")
    monkeypatch.setenv("RAG_AGENT_MODEL_MAX_RETRIES", "5")
    monkeypatch.setenv("RAG_AGENT_MODEL_TIMEOUT_SECONDS", "12.5")

    settings = Settings(_env_file=None)

    assert settings.qwen_chat_model == "qwen-max"
    assert settings.model_max_retries == 5
    assert settings.model_timeout_seconds == 12.5


def test_invalid_model_settings_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_timeout_seconds=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_max_retries=-1)
