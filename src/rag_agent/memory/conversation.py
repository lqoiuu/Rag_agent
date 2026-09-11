"""Conversation bookkeeping: thread ownership, the prompt window and preferences.

Two kinds of memory live here, and the difference matters:

短期记忆
    The messages of one thread. They are *fully* persisted by the checkpointer;
    only the last :data:`DEFAULT_WINDOW_SIZE` of them are rendered into the
    prompt. The window is a cost control, not a storage limit.

长期记忆
    Preferences that belong to a user rather than to one conversation. They are
    stored per ``user_id`` and survive clearing a thread, because "this customer
    prefers email" is not a property of one chat.

Thread ownership is kept as application state, not inside the checkpointer:
LangGraph treats a ``thread_id`` as opaque, so the binding "this conversation
belongs to U1001" has to be enforced by code that knows about users.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from rag_agent.memory.schema import CONVERSATION_SCHEMA
from rag_agent.providers.base import ChatMessage
from rag_agent.storage.sqlite import utc_now

DEFAULT_WINDOW_SIZE = 8
CONVERSATION_ROLES = ("user", "assistant")


class ThreadOwnershipError(RuntimeError):
    """Raised when a thread is used by someone other than its owner."""

    code = "thread_ownership_conflict"


@dataclass(frozen=True, slots=True)
class ThreadSummary:
    """One line of ``rag-agent thread list``."""

    thread_id: str
    user_id: str | None
    created_at: str
    updated_at: str
    message_count: int
    checkpoint_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "checkpoint_count": self.checkpoint_count,
        }


@dataclass(frozen=True, slots=True)
class ConversationWindow:
    """The slice of history that is allowed into one prompt."""

    messages: tuple[ChatMessage, ...]

    @property
    def is_empty(self) -> bool:
        return not self.messages

    def as_dict(self) -> dict[str, object]:
        return {
            "window_size": len(self.messages),
            "roles": [message.role for message in self.messages],
        }


def trim_messages(
    messages: object,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> tuple[ChatMessage, ...]:
    """Keep the newest ``window_size`` messages, preserving their order.

    ``messages`` arrives straight out of a checkpoint, so anything that is not a
    well-formed ``{"role", "content"}`` record is dropped instead of crashing a
    conversation that was persisted by an older version of this project.
    """

    if not isinstance(messages, list) or window_size <= 0:
        return ()

    window: list[ChatMessage] = []
    for item in messages[-window_size:]:
        role = item.get("role") if isinstance(item, dict) else None
        content = item.get("content") if isinstance(item, dict) else None
        if role in CONVERSATION_ROLES and isinstance(content, str) and content:
            window.append(ChatMessage(role=role, content=content))
    return tuple(window)


def render_prompt_context(
    window: ConversationWindow,
    preferences: dict[str, str] | None = None,
) -> str:
    """Render history plus preferences as a prompt block, or an empty string.

    Returning an empty string when there is nothing to say keeps the first turn's
    prompt byte-identical to the prompt a stateless run would build, so adding
    memory does not silently change single-turn behaviour.
    """

    sections: list[str] = []
    if preferences:
        lines = [f"- {key}：{value}" for key, value in sorted(preferences.items())]
        sections.append("已知的长期偏好：\n" + "\n".join(lines))
    if not window.is_empty:
        lines = [
            f"{'用户' if message.role == 'user' else '助手'}：{message.content}"
            for message in window.messages
        ]
        sections.append("最近对话（供理解指代，不作为事实依据）：\n" + "\n".join(lines))
    if not sections:
        return ""
    return "\n\n".join(sections) + "\n\n"


class ConversationStore:
    """Per-user conversation registry backed by the same SQLite file."""

    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        # See SQLiteCheckpointer: this database is touched from the thread that runs
        # the graph, so the connection must not be bound to its creating thread.
        self._connection = sqlite3.connect(self._path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self.initialize()

    @property
    def path(self) -> str:
        return self._path

    def initialize(self) -> None:
        """Create the tables this store owns.

        The checkpointer creates the same schema because both live in one database
        file, but this store must also work on its own — ``rag-agent thread list``
        should not depend on a checkpoint having ever been written. ``IF NOT
        EXISTS`` makes the overlap harmless.
        """

        for statement in CONVERSATION_SCHEMA:
            self._connection.execute(statement)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ConversationStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ threads

    def ensure_thread(self, thread_id: str, user_id: str | None) -> None:
        """Register the thread, or refuse if another user owns it."""

        owner = self.owner_of(thread_id)
        if owner is not None and (user_id or None) != owner:
            raise ThreadOwnershipError(
                f"thread {thread_id!r} belongs to another user; use your own thread id or a new one"
            )

        now = utc_now()
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO _threads (thread_id, user_id, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (thread_id) DO UPDATE SET
                    user_id = COALESCE(_threads.user_id, excluded.user_id),
                    updated_at = excluded.updated_at
                """,
                (thread_id, user_id or None, now, now),
            )

    def touch_thread(self, thread_id: str) -> None:
        """Record that a turn just happened on this thread."""

        with self._connection:
            self._connection.execute(
                "UPDATE _threads SET updated_at = ? WHERE thread_id = ?",
                (utc_now(), thread_id),
            )

    def owner_of(self, thread_id: str) -> str | None:
        row = self._connection.execute(
            "SELECT user_id FROM _threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        if row is None or row["user_id"] is None:
            return None
        return str(row["user_id"])

    def list_threads(self, user_id: str | None = None) -> list[ThreadSummary]:
        """List the caller's threads, plus threads not yet bound to anyone."""

        clause = "" if user_id is None else "WHERE t.user_id = ? OR t.user_id IS NULL"
        params: tuple[object, ...] = () if user_id is None else (user_id,)
        rows = self._connection.execute(
            f"""
            SELECT t.thread_id, t.user_id, t.created_at, t.updated_at,
                   (SELECT COUNT(*) FROM checkpoints c
                     WHERE c.thread_id = t.thread_id) AS checkpoint_count,
                   (SELECT COUNT(*) FROM checkpoint_blobs b
                     WHERE b.thread_id = t.thread_id AND b.channel = 'messages')
                     AS message_count
            FROM _threads t
            {clause}
            ORDER BY t.updated_at DESC, t.thread_id
            """,
            params,
        ).fetchall()
        return [
            ThreadSummary(
                thread_id=str(row["thread_id"]),
                user_id=None if row["user_id"] is None else str(row["user_id"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                message_count=int(row["message_count"]),
                checkpoint_count=int(row["checkpoint_count"]),
            )
            for row in rows
        ]

    def clear_thread(self, thread_id: str) -> int:
        """Delete one conversation's checkpoints and return how many were removed."""

        checkpoint_count = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread_id,)
            ).fetchone()[0]
        )
        with self._connection:
            self._connection.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,)
            )
            self._connection.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = ?", (thread_id,)
            )
            self._connection.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            self._connection.execute("DELETE FROM _threads WHERE thread_id = ?", (thread_id,))
        return checkpoint_count

    # -------------------------------------------------------------- preferences

    def save_preference(self, user_id: str, key: str, value: str) -> None:
        """Store one long-term preference; clearing a thread does not touch it."""

        if not key.strip():
            raise ValueError("preference key must not be empty")
        with self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO user_preferences
                    (user_id, preference_key, preference_value, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, key.strip(), value, utc_now()),
            )

    def load_preferences(self, user_id: str | None) -> dict[str, str]:
        if not user_id:
            return {}
        rows = self._connection.execute(
            """
            SELECT preference_key, preference_value FROM user_preferences
            WHERE user_id = ? ORDER BY preference_key
            """,
            (user_id,),
        ).fetchall()
        return {str(row["preference_key"]): str(row["preference_value"]) for row in rows}


__all__ = [
    "CONVERSATION_ROLES",
    "DEFAULT_WINDOW_SIZE",
    "ConversationStore",
    "ConversationWindow",
    "ThreadOwnershipError",
    "ThreadSummary",
    "render_prompt_context",
    "trim_messages",
]
