"""Environment-backed application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_agent.domain.retrieval import DEFAULT_THRESHOLD, DEFAULT_TOP_K
from rag_agent.providers.qwen import (
    DEFAULT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Validated settings with project-root-relative local paths."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_AGENT_",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    data_dir: Path = Path("data")
    chroma_dir: Path = Path("data/chroma")
    sqlite_path: Path = Path("data/rag_agent.sqlite3")
    qwen_api_key: SecretStr | None = None
    qwen_base_url: str = Field(default=DEFAULT_BASE_URL, min_length=1)
    qwen_chat_model: str = Field(default=DEFAULT_CHAT_MODEL, min_length=1)
    qwen_embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, min_length=1)
    model_timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, gt=0)
    model_max_retries: int = Field(default=DEFAULT_MAX_RETRIES, ge=0)
    retrieval_top_k: int = Field(default=DEFAULT_TOP_K, ge=1)
    retrieval_threshold: float = Field(default=DEFAULT_THRESHOLD, ge=-1.0, le=1.0)

    @field_validator("data_dir", "chroma_dir", "sqlite_path", mode="before")
    @classmethod
    def resolve_project_path(cls, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (PROJECT_ROOT / path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one immutable-by-convention settings instance per process."""

    return Settings()
