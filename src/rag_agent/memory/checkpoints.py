"""SQLite checkpoint saver and conversation registry.

LangGraph can hand its state to any :class:`BaseCheckpointSaver`. The saver here
writes that state to SQLite with the standard library, so a conversation survives
a process restart without adding a dependency, and the on-disk format stays
readable from this project alone.

Three tables mirror the three things LangGraph asks a saver to store:

``checkpoints``
    One row per super-step. ``channel_values`` is *not* stored here: it lives in
    ``checkpoint_blobs`` keyed by channel version, so an unchanged channel is
    written once instead of on every step. The row keeps the parent pointer, which
    is what makes history walkable.
``checkpoint_blobs``
    Serialized channel values, keyed by ``(thread, namespace, channel, version)``.
    The version is part of the key, which is why re-writing an old checkpoint is a
    no-op rather than a rollback.
``checkpoint_writes``
    Per-task writes, including the reserved ``__interrupt__`` channel. Order is
    preserved by ``idx``: the reserved channels use negative indices so they sort
    after ordinary writes.

The table layout itself lives in :mod:`rag_agent.memory.schema`, because the
conversation registry creates the same tables.
"""

from __future__ import annotations

import builtins
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from rag_agent.memory.schema import CONVERSATION_SCHEMA
from rag_agent.storage.sqlite import utc_now


class SQLiteCheckpointer(BaseCheckpointSaver[str]):
    """A synchronous SQLite checkpoint saver.

    The connection is owned by this object, so one saver is one open database.
    Writes are committed per method call, which is what makes "the process died
    mid-conversation" a recoverable situation rather than a lost one.
    """

    def __init__(self, path: Path | str, *, serde: SerializerProtocol | None = None) -> None:
        super().__init__(serde=serde)
        self._path = str(path)
        # LangGraph persists checkpoints from its own thread pool, so this saver is
        # used by more than one thread. ``check_same_thread=False`` allows that, and
        # ``_lock`` is what makes it safe: every transaction is serialized through it
        # (see ``_transaction``).
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self._path, check_same_thread=False, uri=True)
        self._connection.row_factory = sqlite3.Row
        self.initialize()

    @property
    def path(self) -> str:
        return self._path

    def initialize(self) -> None:
        """Create every table; safe to run on every open."""

        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        for statement in CONVERSATION_SCHEMA:
            self._connection.execute(statement)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SQLiteCheckpointer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ reading

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Return one checkpoint: the exact one asked for, or the latest."""

        with self._lock:
            return self._get_tuple(config)

    def _get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        checkpoint_id = get_checkpoint_id(config)

        if checkpoint_id is None:
            row = self._connection.execute(
                """
                SELECT * FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ?
                ORDER BY checkpoint_id DESC LIMIT 1
                """,
                (thread_id, checkpoint_ns),
            ).fetchone()
        else:
            row = self._connection.execute(
                """
                SELECT * FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                """,
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchone()
        if row is None:
            return None

        checkpoint: Checkpoint = cast(
            "Checkpoint", self.serde.loads_typed((row["type"], row["checkpoint"]))
        )
        stored_id = str(row["checkpoint_id"])
        writes = self._load_writes(thread_id, checkpoint_ns, stored_id)
        stored: Checkpoint = {
            **checkpoint,
            "channel_values": self._load_blobs(
                thread_id, checkpoint_ns, checkpoint["channel_versions"]
            ),
        }
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": stored_id,
                }
            },
            checkpoint=stored,
            metadata=cast(
                "CheckpointMetadata",
                self.serde.loads_typed((row["metadata_type"], row["metadata"])),
            ),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": str(row["parent_checkpoint_id"]),
                    }
                }
                if row["parent_checkpoint_id"]
                else None
            ),
            pending_writes=writes,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Yield checkpoints newest first, following the parent chain."""

        with self._lock:
            yield from self._list(config, filter=filter, before=before, limit=limit)

    def _list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id = str(config["configurable"]["thread_id"]) if config else None
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", "")) if config else None
        before_id = get_checkpoint_id(before) if before else None

        target_id = get_checkpoint_id(config) if config else None
        rows: builtins.list[sqlite3.Row]
        if target_id is None:
            where: builtins.list[str] = []
            params: builtins.list[object] = []
            if thread_id is not None:
                where.append("thread_id = ?")
                params.append(thread_id)
            if checkpoint_ns is not None:
                where.append("checkpoint_ns = ?")
                params.append(checkpoint_ns)
            clause = f"WHERE {' AND '.join(where)}" if where else ""
            rows = self._connection.execute(
                f"SELECT * FROM checkpoints {clause} ORDER BY checkpoint_id DESC",
                params,
            ).fetchall()
        else:
            rows = self._walk_parent_chain(thread_id or "", checkpoint_ns or "", target_id)

        yielded = 0
        for row in rows:
            if filter and not _matches_filter(self.serde, row["metadata"], filter):
                continue
            if before_id is not None and str(row["checkpoint_id"]) >= before_id:
                continue
            if limit is not None and yielded >= limit:
                return
            yielded += 1
            loaded = self.get_tuple(
                {
                    "configurable": {
                        "thread_id": str(row["thread_id"]),
                        "checkpoint_ns": str(row["checkpoint_ns"]),
                        "checkpoint_id": str(row["checkpoint_id"]),
                    }
                }
            )
            if loaded is not None:
                yield loaded

    # ------------------------------------------------------------------ writing

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store the channel values this step changed, then the checkpoint row."""

        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        parent_id = config["configurable"].get("checkpoint_id")

        payload: dict[str, Any] = dict(checkpoint)
        values = cast("dict[str, Any]", payload.pop("channel_values", {}))
        stored = self.serde.dumps_typed(payload)
        meta = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))

        with self._transaction():
            for channel, version in new_versions.items():
                typed = (
                    self.serde.dumps_typed(values[channel]) if channel in values else ("empty", b"")
                )
                self._connection.execute(
                    """
                    INSERT OR REPLACE INTO checkpoint_blobs
                        (thread_id, checkpoint_ns, channel, version, type, value)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (thread_id, checkpoint_ns, channel, str(version), typed[0], typed[1]),
                )
            self._connection.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                    (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                     type, checkpoint, metadata_type, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    str(checkpoint["id"]),
                    str(parent_id) if parent_id else None,
                    stored[0],
                    stored[1],
                    meta[0],
                    meta[1],
                    utc_now(),
                ),
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": str(checkpoint["id"]),
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store per-task writes. Reserved channels keep their negative index."""

        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        checkpoint_id = str(config["configurable"]["checkpoint_id"])

        with self._transaction():
            # The duplicate check reads and writes the same rows, so it belongs inside
            # the same transaction; checking first and writing afterwards would let a
            # concurrent task for the same checkpoint slip between the two.
            existing = {
                int(row["idx"])
                for row in self._connection.execute(
                    """
                    SELECT idx FROM checkpoint_writes
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                      AND task_id = ?
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id, task_id),
                ).fetchall()
            }
            for position, (channel, value) in enumerate(writes):
                idx = WRITES_IDX_MAP.get(channel, position)
                if idx >= 0 and idx in existing:
                    continue
                typed = self.serde.dumps_typed(value)
                self._connection.execute(
                    """
                    INSERT OR REPLACE INTO checkpoint_writes
                        (thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                         channel, type, value, task_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        idx,
                        channel,
                        typed[0],
                        typed[1],
                        task_path,
                    ),
                )

    def delete_thread(self, thread_id: str) -> None:
        """Remove every checkpoint, blob and write for one conversation."""

        with self._transaction():
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints", "_threads"):
                self._connection.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))

    # ------------------------------------------------------------------ helpers

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Own the write transaction explicitly and serialize access to it.

        Two facts about how LangGraph drives a synchronous graph shape this code:

        1. ``graph.invoke`` runs nodes 鈥?and therefore checkpoint writes 鈥?on its
           own thread pool, so *several threads* can call ``put``/``put_writes`` on
           one saver at the same time. ``check_same_thread=False`` allows that, but
           Python's ``sqlite3`` module is not safe under concurrent use of one
           connection, so a re-entrant lock serializes every transaction here.
        2. A connection with an open transaction cannot start another one, and a
           plain ``SELECT`` opens a deferred transaction in Python's ``sqlite3``.
           Committing first and then taking the write lock keeps the two from
           meeting.

        The lock is re-entrant so that a read helper called from inside a write
        transaction does not deadlock against its own caller.
        """

        with self._lock:
            if self._connection.in_transaction:
                self._connection.commit()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._connection.rollback()
                raise
            self._connection.commit()

    def _load_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for channel, version in versions.items():
            row = self._connection.execute(
                """
                SELECT type, value FROM checkpoint_blobs
                WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ? AND version = ?
                """,
                (thread_id, checkpoint_ns, channel, str(version)),
            ).fetchone()
            if row is None or row["type"] == "empty":
                continue
            loaded[str(channel)] = self.serde.loads_typed((row["type"], row["value"]))
        return loaded

    def _load_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> builtins.list[tuple[str, str, Any]]:
        rows = self._connection.execute(
            """
            SELECT task_id, channel, type, value FROM checkpoint_writes
            WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            ORDER BY task_id, idx
            """,
            (thread_id, checkpoint_ns, checkpoint_id),
        ).fetchall()
        return [
            (
                str(row["task_id"]),
                str(row["channel"]),
                self.serde.loads_typed((row["type"], row["value"])),
            )
            for row in rows
        ]

    def _walk_parent_chain(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> builtins.list[sqlite3.Row]:
        chain: builtins.list[sqlite3.Row] = []
        current: str | None = checkpoint_id
        while current:
            row = self._connection.execute(
                """
                SELECT * FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                """,
                (thread_id, checkpoint_ns, current),
            ).fetchone()
            if row is None:
                break
            chain.append(row)
            current = str(row["parent_checkpoint_id"]) if row["parent_checkpoint_id"] else None
        return chain


def _matches_filter(
    serde: SerializerProtocol,
    metadata: bytes,
    filter: dict[str, Any],
) -> bool:
    loaded = serde.loads_typed(("json", metadata))
    return all(loaded.get(key) == value for key, value in filter.items())


__all__ = ["CONVERSATION_SCHEMA", "SQLiteCheckpointer"]
