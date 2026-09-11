"""The SQLite checkpoint saver: persistence, history and thread deletion."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph import END, START, StateGraph
from memory_test_support import memory_database_uri

from rag_agent.memory import SQLiteCheckpointer

THREAD = {"configurable": {"thread_id": "T1"}}


class CounterState(TypedDict, total=False):
    """A deliberately small state, to show the saver is not tied to the agent."""

    counter: int
    notes: Annotated[list[str], operator.add]


def _build_counter_graph(checkpointer: SQLiteCheckpointer) -> Any:
    """A two-node graph whose state is easy to assert on."""

    def add_one(state: CounterState) -> dict[str, Any]:
        return {"counter": state.get("counter", 0) + 1, "notes": ["add"]}

    def double(state: CounterState) -> dict[str, Any]:
        return {"counter": state["counter"] * 2, "notes": ["double"]}

    def mark(state: CounterState) -> dict[str, Any]:
        return {"notes": ["mark"]}

    builder: Any = StateGraph(CounterState)
    builder.add_node("add", add_one)
    builder.add_node("double", double)
    builder.add_node("mark", mark)
    builder.add_edge(START, "add")
    builder.add_edge("add", "double")
    builder.add_edge("double", "mark")
    builder.add_edge("mark", END)
    return builder.compile(checkpointer=checkpointer)


def test_checkpoint_survives_reopening_the_database(tmp_path: Path) -> None:
    """The stage 11 claim: a second connection sees the first one's conversation.

    This one uses a real file on purpose. A shared-cache in-memory database is
    destroyed when its last connection closes, so it cannot express "the process
    restarted"; a file can, and it is the honest version of the acceptance test.
    """

    database = tmp_path / "conversation.sqlite3"
    with SQLiteCheckpointer(database) as first:
        graph = _build_counter_graph(first)
        result = graph.invoke({"counter": 1, "notes": []}, config=THREAD)
        assert result["counter"] == 4

    with SQLiteCheckpointer(database) as second:
        graph = _build_counter_graph(second)
        snapshot = graph.get_state(THREAD)

        assert snapshot.values["counter"] == 4
        assert snapshot.values["notes"] == ["add", "double", "mark"]


def test_threads_are_isolated_from_each_other() -> None:
    uri = memory_database_uri()
    with SQLiteCheckpointer(uri) as saver:
        graph = _build_counter_graph(saver)
        graph.invoke({"counter": 1, "notes": []}, config={"configurable": {"thread_id": "A"}})
        graph.invoke({"counter": 10, "notes": []}, config={"configurable": {"thread_id": "B"}})

        assert graph.get_state({"configurable": {"thread_id": "A"}}).values["counter"] == 4
        assert graph.get_state({"configurable": {"thread_id": "B"}}).values["counter"] == 22


def test_history_is_walkable_newest_first() -> None:
    uri = memory_database_uri()
    with SQLiteCheckpointer(uri) as saver:
        graph = _build_counter_graph(saver)
        graph.invoke({"counter": 1, "notes": []}, config=THREAD)

        history = list(saver.list(THREAD))
        counters = [tuple(item.checkpoint["channel_values"].get("notes", ())) for item in history]

        assert counters[0] == ("add", "double", "mark")  # 最新在前
        assert counters[-1] == ()  # 最早的是输入检查点
        # 父链可以逐级回退
        assert history[0].parent_config is not None
        parent = saver.get_tuple(history[0].parent_config)
        assert parent is not None
        assert tuple(parent.checkpoint["channel_values"].get("notes", ())) == ("add", "double")


def test_exact_checkpoint_id_returns_that_checkpoint() -> None:
    uri = memory_database_uri()
    with SQLiteCheckpointer(uri) as saver:
        graph = _build_counter_graph(saver)
        graph.invoke({"counter": 1, "notes": []}, config=THREAD)

        latest = saver.get_tuple(THREAD)
        assert latest is not None
        exact = saver.get_tuple(latest.config)

        assert exact is not None
        assert exact.checkpoint["id"] == latest.checkpoint["id"]


def test_unknown_thread_returns_nothing() -> None:
    uri = memory_database_uri()
    with SQLiteCheckpointer(uri) as saver:
        assert saver.get_tuple({"configurable": {"thread_id": "missing"}}) is None
        assert list(saver.list({"configurable": {"thread_id": "missing"}})) == []


def test_delete_thread_removes_checkpoints_and_blobs() -> None:
    uri = memory_database_uri()
    with SQLiteCheckpointer(uri) as saver:
        graph = _build_counter_graph(saver)
        graph.invoke({"counter": 1, "notes": []}, config=THREAD)
        assert list(saver.list(THREAD)) != []

        saver.delete_thread("T1")

        assert list(saver.list(THREAD)) == []
        assert saver.get_tuple(THREAD) is None
        blobs = saver._connection.execute("SELECT COUNT(*) FROM checkpoint_blobs").fetchone()[0]
        writes = saver._connection.execute("SELECT COUNT(*) FROM checkpoint_writes").fetchone()[0]
        assert blobs == 0
        assert writes == 0


def test_put_is_idempotent_for_the_same_checkpoint() -> None:
    """Writing the same checkpoint twice must not duplicate or roll history back."""

    uri = memory_database_uri()
    with SQLiteCheckpointer(uri) as saver:
        config = {"configurable": {"thread_id": "T1", "checkpoint_ns": "", "checkpoint_id": None}}
        checkpoint = empty_checkpoint()
        metadata: dict[str, Any] = {"source": "input", "step": -1}

        first = saver.put(config, checkpoint, metadata, {"counter": 1})
        saver.put(first, checkpoint, metadata, {"counter": 1})

        rows = saver._connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        assert rows == 1


def test_list_respects_limit_and_before() -> None:
    uri = memory_database_uri()
    with SQLiteCheckpointer(uri) as saver:
        graph = _build_counter_graph(saver)
        graph.invoke({"counter": 1, "notes": []}, config=THREAD)
        graph.invoke({"counter": 5, "notes": []}, config=THREAD)

        limited = list(saver.list(THREAD, limit=2))
        assert len(limited) == 2

        newest = next(iter(saver.list(THREAD)))
        older = list(saver.list(THREAD, before=newest.config))
        assert all(item.checkpoint["id"] != newest.checkpoint["id"] for item in older)
