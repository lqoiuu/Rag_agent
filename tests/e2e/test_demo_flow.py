"""Offline end-to-end proof of the complete demonstration workflow.

The test crosses the real ingestion pipeline, Chroma index, retriever, LangGraph,
SQLite checkpointer and business repository. Only the two external model providers
are deterministic fakes, so CI needs neither a network connection nor a secret.
"""

from __future__ import annotations

import json
from contextlib import ExitStack

from rag_agent.agent import (
    STATUS_ANSWERED,
    STATUS_PENDING_CONFIRMATION,
    STATUS_TICKET_CREATED,
    build_agent_graph,
    chat_turn,
)
from rag_agent.ingestion import INDEXED, ingest_path
from rag_agent.memory import SQLiteCheckpointer
from rag_agent.providers.fake import FakeChatModel, FakeEmbeddingModel
from rag_agent.retrieval import Retriever
from rag_agent.storage import BusinessRepository, MetadataStore
from rag_agent.vectorstore import ChunkVectorStore


def intent_reply(intent: str) -> str:
    return json.dumps(
        {"intent": intent, "confidence": 0.9, "reason": "scripted e2e"},
        ensure_ascii=False,
    )


def answer_reply(answer: str) -> str:
    return json.dumps(
        {"status": "answered", "answer": answer, "citations": [1], "reason": ""},
        ensure_ascii=False,
    )


def extract_reply(*, device_id: str, issue: str, contact: str) -> str:
    return json.dumps(
        {"device_id": device_id, "issue": issue, "contact": contact},
        ensure_ascii=False,
    )


def test_document_to_answer_to_confirmed_ticket(tmp_path) -> None:
    """Run the portfolio demo through one real, persisted application graph."""

    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    manual = raw_dir / "manual.md"
    manual.write_text(
        "# DBX23 产品手册\n\n"
        "## 产品参数\n\n"
        "产品型号为 DBX23，主机额定输入为 20V 2A，充电时间约 6.5 小时。\n",
        encoding="utf-8",
    )

    database = data_dir / "rag_agent.sqlite3"
    embeddings = FakeEmbeddingModel(dimension=16)
    vectors = ChunkVectorStore(data_dir / "chroma")
    model = FakeChatModel(
        [
            intent_reply("knowledge"),
            answer_reply("产品型号为 DBX23。[1]"),
            intent_reply("device"),
            intent_reply("ticket"),
            extract_reply(
                device_id="D2001",
                issue="主刷一直卡住",
                contact="138****0001",
            ),
        ]
    )

    with ExitStack() as stack:
        metadata = stack.enter_context(MetadataStore(database))
        repository = stack.enter_context(BusinessRepository(database))
        checkpointer = stack.enter_context(SQLiteCheckpointer(database))
        repository.seed_from_file("data/business/seed.json")

        indexed = ingest_path(
            manual,
            root=raw_dir,
            store=metadata,
            vectors=vectors,
            embedding_model=embeddings,
        )
        assert indexed.status == INDEXED
        assert indexed.chunk_count > 0

        graph = build_agent_graph(
            retriever=Retriever(vectors=vectors, embedding_model=embeddings),
            chat_model=model,
            repository=repository,
            checkpointer=checkpointer,
        )
        thread_id = "T-E2E-DEMO"

        knowledge = chat_turn(
            graph,
            "产品型号是什么？",
            thread_id=thread_id,
            user_id="U1001",
        )
        assert knowledge.status == STATUS_ANSWERED
        assert knowledge.run.answer == "产品型号为 DBX23。[1]"
        assert knowledge.run.citations
        assert knowledge.run.citations[0]["source"] == "manual.md"

        device = chat_turn(
            graph,
            "D2002 还在保修吗？",
            thread_id=thread_id,
            user_id="U1001",
        )
        assert device.status == STATUS_ANSWERED
        assert device.turn_tool_results[0]["tool"] == "device.lookup"
        assert "2028-01-10" in device.run.answer

        pending = chat_turn(
            graph,
            "帮我把 D2001 报修，主刷一直卡住，联系我 138****0001",
            thread_id=thread_id,
            user_id="U1001",
        )
        assert pending.status == STATUS_PENDING_CONFIRMATION
        assert pending.paused is True
        assert repository.list_tickets() == ()

        confirmed = chat_turn(
            graph,
            "",
            thread_id=thread_id,
            resume={"confirmed": True},
        )
        assert confirmed.status == STATUS_TICKET_CREATED
        assert confirmed.paused is False
        assert len(repository.list_tickets()) == 1
        assert confirmed.run.created_ticket is not None
        assert str(confirmed.run.created_ticket["ticket_id"]).startswith("T")
        result_data = confirmed.turn_tool_results[0]["data"]
        assert isinstance(result_data, dict)
        assert result_data["created"] is True
