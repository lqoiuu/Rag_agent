"""Multi-turn conversation: memory, isolation and the resumable confirmation gate."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
from typing import Any

import pytest
from agent_test_support import (
    MANUAL_PAGE_27,
    agent_dependencies,
    answer_reply,
    extract_reply,
    intent_reply,
)
from memory_test_support import memory_database_uri

from rag_agent.agent import (
    STATUS_ANSWERED,
    STATUS_PENDING_CONFIRMATION,
    STATUS_TICKET_CANCELLED,
    STATUS_TICKET_CREATED,
    build_agent_graph,
    chat_turn,
)
from rag_agent.memory import SQLiteCheckpointer
from rag_agent.providers.fake import FakeChatModel
from rag_agent.storage.business import BusinessRepository


@pytest.fixture
def conversations() -> Iterator[Any]:
    """A graph factory whose retriever, model, store and checkpointer stay open.

    One checkpointer per factory call, so each test's threads are isolated, and the
    exit stack tears everything down once the test has finished asserting — the
    repository must still be open when a test checks what was written.
    """

    with ExitStack() as stack:

        def open_graph(*replies: str) -> tuple[Any, FakeChatModel, BusinessRepository]:
            retriever, model, repository = stack.enter_context(agent_dependencies(*replies))
            saver = stack.enter_context(SQLiteCheckpointer(memory_database_uri()))
            graph = build_agent_graph(
                retriever=retriever,
                chat_model=model,
                repository=repository,
                checkpointer=saver,
            )
            return graph, model, repository

        yield open_graph


def test_second_turn_reads_the_first_turn_message(conversations: Any) -> None:
    """The window is what makes a follow-up question understandable."""

    graph, model, _repository = conversations(
        intent_reply("knowledge"),
        answer_reply(MANUAL_PAGE_27),
        intent_reply("knowledge"),
        answer_reply(MANUAL_PAGE_27),
    )
    first = chat_turn(graph, "D2002 还在保修吗", thread_id="T1", user_id="U1001")
    graph.update_state(
        {"configurable": {"thread_id": "T1"}},
        {
            "messages": [
                {"role": "user", "content": "D2002 还在保修吗"},
                {"role": "assistant", "content": first.run.answer},
            ]
        },
    )

    second = chat_turn(graph, "那它的型号是什么", thread_id="T1", user_id="U1001")

    assert second.run.status == STATUS_ANSWERED
    # 第二轮把第一轮的两条消息放进 Prompt：这就是「多轮」的实际证据
    prompt = model.calls[-1][-1].content
    assert "D2002 还在保修吗" in prompt
    assert "用户：" in prompt and "助手：" in prompt


def test_window_trims_but_the_checkpoint_keeps_everything(conversations: Any) -> None:
    graph, _model, _repository = conversations(
        intent_reply("knowledge"), answer_reply(MANUAL_PAGE_27)
    )
    graph.update_state(
        {"configurable": {"thread_id": "T-window"}},
        {"messages": [{"role": "user", "content": f"历史第{index}条"} for index in range(1, 11)]},
    )

    turn = chat_turn(graph, "新的问题", thread_id="T-window", user_id="U1001", window_size=3)

    assert len(turn.window.messages) == 3
    assert [message.content for message in turn.window.messages] == [
        "历史第8条",
        "历史第9条",
        "历史第10条",
    ]

    stored = graph.get_state({"configurable": {"thread_id": "T-window"}})
    assert len(stored.values["messages"]) == 10  # 库里还是全量


def test_threads_do_not_share_context(conversations: Any) -> None:
    graph, model, _repository = conversations(
        intent_reply("knowledge"), answer_reply(MANUAL_PAGE_27)
    )
    graph.update_state(
        {"configurable": {"thread_id": "A"}},
        {"messages": [{"role": "user", "content": "线程A的秘密"}]},
    )

    turn_b = chat_turn(graph, "你好", thread_id="B", user_id="U1001")

    assert turn_b.window.is_empty
    assert "线程A的秘密" not in model.calls[-1][-1].content


def test_cancelling_a_pending_action_writes_nothing(conversations: Any) -> None:
    graph, _model, repository = conversations(
        intent_reply("ticket"),
        extract_reply(device_id="D2001", issue="主刷一直卡住", contact="138****0001"),
    )
    paused = chat_turn(
        graph,
        "帮我把 D2001 报修，主刷一直卡住，联系我 138****0001",
        thread_id="T1",
        user_id="U1001",
    )
    assert paused.run.status == STATUS_PENDING_CONFIRMATION
    assert repository.list_tickets() == ()

    cancelled = chat_turn(graph, "", thread_id="T1", resume={"confirmed": False})

    assert cancelled.run.status == STATUS_TICKET_CANCELLED
    assert cancelled.run.created_ticket is None
    assert cancelled.paused is False
    assert repository.list_tickets() == ()
    assert "ticket_create:cancelled" in cancelled.run.trace


def test_a_second_resume_cannot_write_twice(conversations: Any) -> None:
    """Once the pause is resolved the graph has nothing left to resume."""

    graph, _model, repository = conversations(
        intent_reply("ticket"),
        extract_reply(device_id="D2001", issue="主刷一直卡住", contact="138****0001"),
    )
    chat_turn(
        graph,
        "帮我把 D2001 报修，主刷一直卡住，联系我 138****0001",
        thread_id="T1",
        user_id="U1001",
    )
    first = chat_turn(graph, "", thread_id="T1", resume={"confirmed": True})
    second = chat_turn(graph, "", thread_id="T1", resume={"confirmed": True})

    assert first.run.status == STATUS_TICKET_CREATED
    assert second.paused is False
    assert len(repository.list_tickets()) == 1


def test_device_turn_resolves_the_device_from_the_message(conversations: Any) -> None:
    """The device id must come from the message, not only from a CLI flag.

    Without this the node listed every device of the user instead of answering about
    the one that was named.
    """

    graph, _model, _repository = conversations(intent_reply("device"))

    turn = chat_turn(graph, "D2002 还在保修吗", thread_id="T1", user_id="U1001")

    assert turn.run.status == STATUS_ANSWERED
    payload = turn.run.tool_results[0]["data"]
    assert isinstance(payload, dict)
    device = payload["device"]
    assert isinstance(device, dict)
    assert device["device_id"] == "D2002"


def test_follow_up_keeps_working_once_history_is_in_the_prompt(conversations: Any) -> None:
    """The regression that motivated resolving the device id deterministically.

    Measured with the real model: the same follow-up was classified as a device
    question with no history and as a knowledge question once one turn of history was
    rendered into the prompt, which turned it into a refusal.
    """

    graph, _model, _repository = conversations(intent_reply("device"), intent_reply("device"))
    first = chat_turn(graph, "D2002 还在保修吗", thread_id="T1", user_id="U1001")
    graph.update_state(
        {"configurable": {"thread_id": "T1"}},
        {
            "messages": [
                {"role": "user", "content": "D2002 还在保修吗"},
                {"role": "assistant", "content": first.run.answer},
            ]
        },
    )

    second = chat_turn(graph, "那它的保修期是多久", thread_id="T1", user_id="U1001")

    assert second.run.status == STATUS_ANSWERED
    assert second.window.messages  # 历史确实进了窗口
    payload = second.run.tool_results[0]["data"]
    assert isinstance(payload, dict)
    device = payload["device"]
    assert isinstance(device, dict)
    assert device["device_id"] == "D2002"


def test_the_corrected_intent_is_visible_in_the_trace(conversations: Any) -> None:
    """A corrected classification must never look like the model's own decision."""

    graph, _model, _repository = conversations(
        intent_reply("device"),
        intent_reply("knowledge"),  # 模型在有历史的线程里把它判成了知识问题
    )
    first = chat_turn(graph, "D2002 还在保修吗", thread_id="T1", user_id="U1001")
    graph.update_state(
        {"configurable": {"thread_id": "T1"}},
        {
            "messages": [
                {"role": "user", "content": "D2002 还在保修吗"},
                {"role": "assistant", "content": first.run.answer},
            ]
        },
    )

    second = chat_turn(graph, "那它的保修期是多久", thread_id="T1", user_id="U1001")

    assert second.run.intent == "device"
    assert "classify:device:rerouted-from-knowledge" in second.run.trace
    assert second.run.status == STATUS_ANSWERED


def test_preferences_reach_the_prompt(conversations: Any) -> None:
    graph, model, _repository = conversations(
        intent_reply("knowledge"), answer_reply(MANUAL_PAGE_27)
    )

    chat_turn(
        graph,
        "制造商地址在哪里",
        thread_id="T1",
        user_id="U1001",
        preferences={"contact": "email"},
    )

    assert "contact：email" in model.calls[-1][-1].content


def test_turn_reports_its_checkpoint_growth(conversations: Any) -> None:
    graph, _model, _repository = conversations(
        intent_reply("knowledge"), answer_reply(MANUAL_PAGE_27)
    )

    turn = chat_turn(graph, "制造商地址在哪里", thread_id="T1", user_id="U1001")

    assert turn.checkpoints_before == 0
    assert turn.checkpoints_after > 0
    assert turn.as_dict()["memory"]["checkpoints_after"] == turn.checkpoints_after


def test_a_turn_reports_only_its_own_writes(conversations: Any) -> None:
    """``run.trace`` accumulates over the thread; the per-turn slice must not.

    A UI that showed the accumulated trace would claim the current turn retried every
    step of the whole conversation.
    """

    graph, _model, _repository = conversations(intent_reply("device"), intent_reply("device"))
    first = chat_turn(graph, "D2002 还在保修吗", thread_id="T1", user_id="U1001")

    second = chat_turn(graph, "D2001 还在保修吗", thread_id="T1", user_id="U1001")

    assert first.turn_trace == ("classify:device", "device:ok")
    # 累积视图保留整条线程
    assert second.run.trace[:2] == ("classify:device", "device:ok")
    assert len(second.run.trace) == 4
    # 本轮视图只有本轮
    assert second.turn_trace == ("classify:device", "device:ok")
    assert len(second.turn_tool_results) == 2  # 本轮 device.lookup + order.lookup
    assert len(second.run.tool_results) == 4  # 整条线程累积


def test_a_resumed_turn_reports_only_the_resume(conversations: Any) -> None:
    """On resume the accumulated trace comes from the snapshot, not a fresh run."""

    graph, _model, _repository = conversations(
        intent_reply("ticket"),
        extract_reply(device_id="D2001", issue="主刷一直卡住", contact="138****0001"),
    )
    paused = chat_turn(
        graph,
        "帮我把 D2001 报修，主刷一直卡住，联系我 138****0001",
        thread_id="T1",
        user_id="U1001",
    )
    assert "ticket_create" not in " ".join(paused.run.trace)

    resumed = chat_turn(graph, "", thread_id="T1", resume={"confirmed": True})

    assert resumed.turn_trace == ("ticket_create:created=True",)
    assert len(resumed.run.trace) > len(resumed.turn_trace)
