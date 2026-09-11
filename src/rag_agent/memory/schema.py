"""SQLite schema for conversation memory.

The tables live here rather than inside either class because both classes create
them: the checkpointer needs them when a graph runs, and the conversation registry
needs them even when no checkpoint has ever been written. Keeping the statements in
one module means the two never disagree about the layout, and neither has to import
the other.
"""

from __future__ import annotations

CONVERSATION_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        checkpoint_id TEXT NOT NULL,
        parent_checkpoint_id TEXT,
        type TEXT,
        checkpoint BLOB NOT NULL,
        metadata_type TEXT,
        metadata BLOB NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS checkpoint_blobs (
        thread_id TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        channel TEXT NOT NULL,
        version TEXT NOT NULL,
        type TEXT NOT NULL,
        value BLOB NOT NULL,
        PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS checkpoint_writes (
        thread_id TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        checkpoint_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        idx INTEGER NOT NULL,
        channel TEXT NOT NULL,
        type TEXT,
        value BLOB NOT NULL,
        task_path TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS _threads (
        thread_id TEXT PRIMARY KEY,
        user_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_preferences (
        user_id TEXT NOT NULL,
        preference_key TEXT NOT NULL,
        preference_value TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (user_id, preference_key)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON checkpoints (thread_id, checkpoint_id)",
    "CREATE INDEX IF NOT EXISTS idx_writes_thread ON checkpoint_writes (thread_id, checkpoint_id)",
)

__all__ = ["CONVERSATION_SCHEMA"]
