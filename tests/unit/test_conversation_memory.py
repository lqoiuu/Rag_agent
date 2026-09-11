"""Thread bookkeeping, the prompt window and long-term preferences."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from memory_test_support import memory_database_uri

from rag_agent.memory import (
    ConversationStore,
    ConversationWindow,
    ThreadOwnershipError,
    render_prompt_context,
    trim_messages,
)
from rag_agent.providers.base import ChatMessage


def test_window_keeps_the_newest_messages_in_order() -> None:
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"m{index}"}
        for index in range(12)
    ]

    window = trim_messages(messages, window_size=4)

    assert [message.content for message in window] == ["m8", "m9", "m10", "m11"]
    assert all(isinstance(message, ChatMessage) for message in window)


def test_window_drops_records_it_cannot_trust() -> None:
    """A checkpoint written by an older version must not break a conversation."""

    messages = [
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": ""},
        {"role": "user", "content": "keep me"},
        "not a mapping",
        {"role": "assistant"},  # 缺 content
    ]

    window = trim_messages(messages, window_size=8)

    assert [message.content for message in window] == ["keep me"]


def test_window_handles_empty_and_zero_size() -> None:
    assert trim_messages(None, window_size=8) == ()
    assert trim_messages([], window_size=8) == ()
    assert trim_messages([{"role": "user", "content": "x"}], window_size=0) == ()


def test_empty_context_renders_as_empty_string() -> None:
    """The first turn must produce the same prompt a stateless run would."""

    assert render_prompt_context(ConversationWindow(messages=())) == ""
    assert render_prompt_context(ConversationWindow(messages=()), {}) == ""


def test_context_labels_the_window_as_non_authoritative() -> None:
    window = ConversationWindow(
        messages=(
            ChatMessage(role="user", content="D2001 还在保修吗"),
            ChatMessage(role="assistant", content="在保。"),
        )
    )

    rendered = render_prompt_context(window, {"contact": "email"})

    assert "长期偏好" in rendered
    assert "contact：email" in rendered
    assert "最近对话" in rendered
    assert "用户：D2001 还在保修吗" in rendered
    assert "助手：在保。" in rendered
    assert "不作为事实依据" in rendered


@pytest.fixture
def store() -> Iterator[ConversationStore]:
    with ConversationStore(memory_database_uri()) as opened:
        yield opened


def test_thread_is_registered_with_its_owner(store: ConversationStore) -> None:
    store.ensure_thread("T1", "U1001")

    assert store.owner_of("T1") == "U1001"
    assert [thread.thread_id for thread in store.list_threads("U1001")] == ["T1"]
    assert store.list_threads("U1002") == []


def test_another_user_cannot_reuse_a_thread(store: ConversationStore) -> None:
    store.ensure_thread("T1", "U1001")

    with pytest.raises(ThreadOwnershipError) as excinfo:
        store.ensure_thread("T1", "U1002")

    assert excinfo.value.code == "thread_ownership_conflict"


def test_unbound_thread_can_be_claimed_once(store: ConversationStore) -> None:
    store.ensure_thread("T1", None)

    assert store.owner_of("T1") is None
    store.ensure_thread("T1", "U1001")
    assert store.owner_of("T1") == "U1001"


def test_preferences_are_per_user_and_survive_clearing_a_thread(
    store: ConversationStore,
) -> None:
    store.ensure_thread("T1", "U1001")
    store.save_preference("U1001", "contact", "email")
    store.save_preference("U1002", "contact", "phone")

    store.clear_thread("T1")

    assert store.owner_of("T1") is None
    assert store.load_preferences("U1001") == {"contact": "email"}
    assert store.load_preferences("U1002") == {"contact": "phone"}


def test_preferences_need_a_non_empty_key(store: ConversationStore) -> None:
    with pytest.raises(ValueError):
        store.save_preference("U1001", "   ", "email")

    assert store.load_preferences("U1001") == {}
    assert store.load_preferences(None) == {}


def test_list_reports_checkpoint_and_message_counts(store: ConversationStore) -> None:
    store.ensure_thread("T1", "U1001")

    summary = store.list_threads("U1001")[0]

    assert summary.message_count == 0
    assert summary.checkpoint_count == 0
    assert summary.created_at
    assert summary.as_dict()["thread_id"] == "T1"
