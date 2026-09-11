"""Shared resources for the Streamlit app.

Everything expensive lives here so the pages stay thin. Three Streamlit rules shape
this module:

* a rerun re-executes the whole script, so objects that are expensive to build (the
  vector store, the HTTP client, the SQLite checkpointer) are created once per process
  and reused — that is what ``st.cache_resource`` is for;
* per-user data must never live at module level, so nothing here is mutable
  conversation state. Threads live in SQLite; the UI's own message list lives in
  ``st.session_state``;
* caches need a bound, so the resource cache carries a TTL and releases its handles
  when the entry is dropped.

The resource holder is a plain dataclass rather than a bag of globals: a page asks for
it explicitly, which keeps the data flow visible and lets tests build one against a
temporary index.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from rag_agent.config import build_chat_model, build_embedding_model, get_settings
from rag_agent.config.settings import Settings
from rag_agent.memory import ConversationStore, SQLiteCheckpointer
from rag_agent.retrieval import Retriever, RetrieverConfig
from rag_agent.storage import BusinessRepository, MetadataStore
from rag_agent.vectorstore import ChunkVectorStore


@dataclass(slots=True)
class AppResources:
    """One process-wide set of connections, models and stores."""

    settings: Settings
    vectors: ChunkVectorStore
    metadata: MetadataStore
    embedding_model: Any
    chat_model: Any
    checkpointer: SQLiteCheckpointer
    conversations: ConversationStore
    repository: BusinessRepository
    retriever: Retriever

    def fingerprint(self) -> dict[str, int | str]:
        """What this process is actually connected to, for the sidebar."""

        return {
            "vector_count": self.vectors.count(),
            "document_count": len(self.metadata.list_documents()),
            "chroma_dir": str(self.settings.chroma_dir),
            "checkpoint_path": str(self.settings.checkpoint_path),
            "chat_model": self.settings.qwen_chat_model,
            "embedding_model": self.settings.qwen_embedding_model,
            "window_size": self.settings.conversation_window_size,
        }

    def close(self) -> None:
        """Release the SQLite handles and model clients."""

        self.conversations.close()
        self.checkpointer.close()
        self.repository.close()
        self.metadata.close()


def build_resources() -> AppResources:
    """Build the full dependency set once. Wrap this in a resource cache."""

    settings = get_settings()
    embedding_model = build_embedding_model(settings)
    chat_model = build_chat_model(settings)
    checkpointer = SQLiteCheckpointer(settings.checkpoint_path)
    conversations = ConversationStore(settings.checkpoint_path)
    repository = BusinessRepository(settings.sqlite_path)
    repository.seed_from_file(settings.data_dir / "business" / "seed.json")
    vectors = ChunkVectorStore(settings.chroma_dir)
    return AppResources(
        settings=settings,
        vectors=vectors,
        metadata=MetadataStore(settings.sqlite_path),
        embedding_model=embedding_model,
        chat_model=chat_model,
        checkpointer=checkpointer,
        conversations=conversations,
        repository=repository,
        retriever=Retriever(
            vectors=vectors,
            embedding_model=embedding_model,
            config=RetrieverConfig(
                top_k=settings.retrieval_top_k,
                threshold=settings.retrieval_threshold,
            ),
        ),
    )


@st.cache_resource(ttl="1h", on_release=lambda resources: resources.close())
def get_resources() -> AppResources:
    """The one cached resource holder for this process.

    ``on_release`` closes the SQLite connections when the cache entry is dropped, so a
    long-running server does not accumulate open file handles.
    """

    return build_resources()


def default_user_id() -> str:
    """The demo identity. This is *not* authentication; the sidebar says so."""

    return "U1001"


def new_thread_id() -> str:
    """A short, sortable, obviously generated conversation id."""

    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"T{stamp}-{uuid.uuid4().hex[:6]}"


def raw_directory(settings: Settings) -> Path:
    """Where uploaded documents are written before ingestion."""

    target = settings.data_dir / "raw"
    target.mkdir(parents=True, exist_ok=True)
    return target


__all__ = [
    "AppResources",
    "build_resources",
    "default_user_id",
    "get_resources",
    "new_thread_id",
    "raw_directory",
]
