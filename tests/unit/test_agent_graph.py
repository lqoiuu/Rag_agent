"""Graph routing, ticket confirmation and the guards around writes."""

from __future__ import annotations

from agent_test_support import (
    MANUAL_PAGE_27,
    agent_dependencies,
    answer_reply,
    extract_reply,
    insufficient_reply,
    intent_reply,
)

from rag_agent.agent import (
    NODE_CLARIFY,
    NODE_CLASSIFY,
    NODE_DEVICE,
    NODE_KNOWLEDGE,
    NODE_TICKET_COLLECT,
    NODE_TICKET_CREATE,
    NODE_TICKET_PENDING,
    STATUS_ANSWERED,
    STATUS_ERROR,
    STATUS_NEEDS_INPUT,
    STATUS_PENDING_CONFIRMATION,
    STATUS_REFUSED,
    STATUS_TICKET_CREATED,
    AgentNodes,
    Intent,
    build_agent_graph,
    run_agent,
)
from rag_agent.tools import create_ticket
from rag_agent.tools.models import CreateTicketArgs


def test_knowledge_question_is_answered_with_citations() -> None:
    with agent_dependencies(
        intent_reply("knowledge"), answer_reply(MANUAL_PAGE_27, citations=[1])
    ) as (retriever, model, repository):
        run = run_agent(
            MANUAL_PAGE_27, retriever=retriever, chat_model=model, repository=repository
        )

    assert run.status == STATUS_ANSWERED
    assert run.intent == Intent.KNOWLEDGE
    assert run.answer == MANUAL_PAGE_27
    assert run.citations[0]["page"] == 27
    assert run.trace[0] == "classify:knowledge"
    assert "knowledge:answered:citations=1" in run.trace


def test_knowledge_refusal_is_reported_not_guessed() -> None:
    with agent_dependencies(intent_reply("knowledge"), insufficient_reply()) as (
        retriever,
        model,
        repository,
    ):
        run = run_agent("保修期多久", retriever=retriever, chat_model=model, repository=repository)

    assert run.status == STATUS_REFUSED
    assert run.answer == ""
    assert run.error_message == "资料未涉及该问题。"
    assert any(entry.startswith("knowledge:refused") for entry in run.trace)


def test_unknown_intent_asks_for_clarification() -> None:
    with agent_dependencies(intent_reply("unknown", 0.9)) as (retriever, model, repository):
        run = run_agent("嗯……", retriever=retriever, chat_model=model, repository=repository)

    assert run.status == STATUS_NEEDS_INPUT
    assert run.intent == Intent.UNKNOWN
    assert run.clarification_turns == 1
    assert "还需要你补充" in run.answer
    assert any(entry.startswith("clarify:ask") for entry in run.trace)


def test_low_confidence_intent_also_clarifies() -> None:
    with agent_dependencies(intent_reply("ticket", 0.1)) as (retriever, model, repository):
        run = run_agent("报修", retriever=retriever, chat_model=model, repository=repository)

    assert run.status == STATUS_NEEDS_INPUT
    assert run.intent == Intent.UNKNOWN


def test_device_question_without_user_id_asks_for_it() -> None:
    with agent_dependencies(intent_reply("device")) as (retriever, model, repository):
        run = run_agent(
            "我的机器还在保修吗", retriever=retriever, chat_model=model, repository=repository
        )

    assert run.status == STATUS_NEEDS_INPUT
    assert "user_id" in run.trace[1]
    assert "U1001" in run.answer  # 提示里给出编号示例


def test_device_question_reports_warranty_from_the_store() -> None:
    with agent_dependencies(intent_reply("device")) as (retriever, model, repository):
        run = run_agent(
            "D2002 还在保修吗",
            retriever=retriever,
            chat_model=model,
            repository=repository,
            user_id="U1001",
            device_id="D2002",
        )

    assert run.status == STATUS_ANSWERED
    assert "已过保" in run.answer or "在保" in run.answer
    assert run.tool_results[0]["tool"] == "device.lookup"
    assert run.tool_results[1]["tool"] == "order.lookup"
    assert "device:ok" in run.trace


def test_device_lookup_for_another_user_fails_cleanly() -> None:
    with agent_dependencies(intent_reply("device")) as (retriever, model, repository):
        run = run_agent(
            "查一下 D2003",
            retriever=retriever,
            chat_model=model,
            repository=repository,
            user_id="U1001",
            device_id="D2003",
        )

    assert run.status == STATUS_ERROR
    assert run.error_code == "permission_denied"
    assert run.tool_results[0]["error_code"] == "permission_denied"


def test_ticket_request_stops_at_the_confirmation_step() -> None:
    with agent_dependencies(
        intent_reply("ticket"),
        extract_reply(device_id="D2001", issue="主刷一直卡住", contact="138****0001"),
    ) as (retriever, model, repository):
        run = run_agent(
            "帮我把 D2001 报修，主刷一直卡住，联系我 138****0001",
            retriever=retriever,
            chat_model=model,
            repository=repository,
            user_id="U1001",
        )

        assert run.status == STATUS_PENDING_CONFIRMATION
        assert run.pending_confirmation is not None
        assert run.pending_confirmation["confirmed"] is False
        assert "确认" in run.answer
        assert repository.list_tickets() == ()
        assert "ticket_create" not in " ".join(run.trace)


def test_ticket_request_missing_fields_asks_for_them() -> None:
    with agent_dependencies(intent_reply("ticket"), extract_reply(issue="主刷卡住")) as (
        retriever,
        model,
        repository,
    ):
        run = run_agent(
            "帮我报修，主刷卡住",
            retriever=retriever,
            chat_model=model,
            repository=repository,
            user_id="U1001",
        )

    assert run.status == STATUS_NEEDS_INPUT
    assert "设备编号" in run.answer
    assert "联系方式" in run.answer


def test_confirmed_run_creates_exactly_one_ticket() -> None:
    with agent_dependencies(
        intent_reply("ticket"),
        extract_reply(device_id="D2001", issue="主刷一直卡住", contact="138****0001"),
    ) as (retriever, model, repository):
        run = run_agent(
            "帮我把 D2001 报修，主刷一直卡住，联系我 138****0001",
            retriever=retriever,
            chat_model=model,
            repository=repository,
            user_id="U1001",
            confirmation={"confirmed": True},
        )

        assert run.status == STATUS_TICKET_CREATED
        assert run.created_ticket is not None
        assert run.created_ticket["ticket_id"].startswith("T")
        assert len(repository.list_tickets()) == 1


def test_clarification_limit_prevents_endless_asking() -> None:
    with agent_dependencies(intent_reply("unknown")) as (retriever, model, repository):
        run = run_agent(
            "嗯",
            retriever=retriever,
            chat_model=model,
            repository=repository,
            clarification_turns=3,
            max_clarifications=3,
        )

    assert run.status == STATUS_ERROR
    assert run.error_code == "clarification_limit"
    assert "clarify:limit" in run.trace


def test_graph_has_no_direct_edge_from_collection_to_the_write() -> None:
    """The acceptance requirement, checked structurally rather than by behaviour."""

    with agent_dependencies() as (retriever, model, repository):
        graph = build_agent_graph(retriever=retriever, chat_model=model, repository=repository)
        edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}

    assert (NODE_TICKET_COLLECT, NODE_TICKET_CREATE) not in edges
    assert (NODE_CLASSIFY, NODE_TICKET_CREATE) not in edges
    assert (NODE_CLARIFY, NODE_TICKET_CREATE) not in edges
    assert (NODE_TICKET_PENDING, NODE_TICKET_CREATE) in edges
    assert (NODE_TICKET_COLLECT, NODE_TICKET_PENDING) in edges
    assert (NODE_CLASSIFY, NODE_KNOWLEDGE) in edges
    assert (NODE_CLASSIFY, NODE_DEVICE) in edges
    assert (NODE_CLASSIFY, NODE_TICKET_COLLECT) in edges
    assert (NODE_CLASSIFY, NODE_CLARIFY) in edges


def test_create_node_refuses_to_write_without_confirmation() -> None:
    with agent_dependencies() as (retriever, model, repository):
        nodes = AgentNodes(retriever=retriever, chat_model=model, repository=repository)
        update = nodes.ticket_create(
            {
                "question": "报修",
                "ticket_draft": {
                    "user_id": "U1001",
                    "device_id": "D2001",
                    "issue": "主刷一直卡住",
                    "contact": "138****0001",
                },
                "confirmation": None,
            }
        )

        assert update["status"] == STATUS_ERROR
        assert update["error_code"] == "confirmation_required"
        assert repository.list_tickets() == ()


def test_tool_layer_also_refuses_unconfirmed_writes() -> None:
    """Two independent barriers: the graph edge and the tool contract."""

    with agent_dependencies() as (_retriever, _model, repository):
        result = create_ticket(
            CreateTicketArgs(
                user_id="U1001",
                device_id="D2001",
                issue="主刷一直卡住",
                contact="138****0001",
            ),
            repository=repository,
        )

        assert result.error_code == "confirmation_required"
        assert repository.list_tickets() == ()
