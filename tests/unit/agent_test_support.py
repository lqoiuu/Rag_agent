"""Shared fixtures for the agent tests: scripted replies and in-memory deps."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from ingestion_test_support import make_vector_store

from rag_agent.providers.fake import FakeChatModel, FakeEmbeddingModel
from rag_agent.retrieval import Retriever
from rag_agent.storage.business import BusinessRepository
from rag_agent.vectorstore import ChunkVectorStore

DIMENSION = 16
MANUAL_PAGE_27 = "产品型号 DBX23 主机额定输入 20V 2A 充电时间约 6.5h"


def intent_reply(intent: str, confidence: float = 0.9) -> str:
    return json.dumps(
        {"intent": intent, "confidence": confidence, "reason": "scripted"},
        ensure_ascii=False,
    )


def answer_reply(text: str, citations: Sequence[int] = (1,)) -> str:
    return json.dumps(
        {
            "status": "answered",
            "answer": text,
            "citations": list(citations),
            "reason": "",
        },
        ensure_ascii=False,
    )


def insufficient_reply(reason: str = "资料未涉及该问题。") -> str:
    return json.dumps(
        {"status": "insufficient", "answer": "", "citations": [], "reason": reason},
        ensure_ascii=False,
    )


def extract_reply(
    *, device_id: str | None = None, issue: str | None = None, contact: str | None = None
) -> str:
    return json.dumps(
        {"device_id": device_id, "issue": issue, "contact": contact}, ensure_ascii=False
    )


def make_retriever(*contents: str) -> Retriever:
    vectors: ChunkVectorStore = make_vector_store()
    model = FakeEmbeddingModel(dimension=DIMENSION)
    if contents:
        from rag_agent.domain.documents import DocumentChunk, build_chunk_id

        chunks = tuple(
            DocumentChunk(
                chunk_id=build_chunk_id("doc-1", index),
                document_id="doc-1",
                source="manual.pdf",
                content=content,
                index=index,
                char_range=(0, len(content)),
                page=27,
            )
            for index, content in enumerate(contents, start=1)
        )
        vectors.upsert(chunks, model.embed([chunk.content for chunk in chunks]).vectors)
    return Retriever(vectors=vectors, embedding_model=model)


@contextmanager
def agent_dependencies(
    *replies: str,
) -> Iterator[tuple[Retriever, FakeChatModel, BusinessRepository]]:
    """Build a retriever, a scripted chat model and a seeded in-memory store."""

    with BusinessRepository(":memory:") as repository:
        repository.seed_from_file("data/business/seed.json")
        yield make_retriever(MANUAL_PAGE_27), FakeChatModel(list(replies)), repository
