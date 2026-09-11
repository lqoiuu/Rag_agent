"""Shared fixtures for the conversation-memory tests.

The tests need a real SQLite database rather than ``:memory:``, because the point
of a checkpointer is that a *second* connection — a restarted process — can read
what the first one wrote. A named shared-cache in-memory database gives exactly
that without touching the filesystem:

* the name is per-test, so tests cannot see each other's state;
* a named memory database survives until its last connection closes, which is
  what lets a test close the writer and reopen it as a "restart";
* nothing is left behind for the next test run.
"""

from __future__ import annotations

import uuid


def memory_database_uri() -> str:
    """A unique in-memory database that several connections can share."""

    return f"file:memory-{uuid.uuid4().hex}?mode=memory&cache=shared"
