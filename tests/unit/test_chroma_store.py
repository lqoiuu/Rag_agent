"""Vector store behaviour over an in-memory Chroma client."""

from __future__ import annotations

import pytest
from ingestion_test_support import make_document_chunks, make_vector_store

from rag_agent.vectorstore import ChunkVectorStore

DOC_A = "doc-aaaa"
DOC_B = "doc-bbbb"


@pytest.fixture
def vectors() -> ChunkVectorStore:
    return make_vector_store()


def test_upsert_stores_chunks_and_metadata(vectors: ChunkVectorStore) -> None:
    chunks = make_document_chunks(DOC_A, "第一段", "第二段")

    written = vectors.upsert(chunks, [[1.0, 0.0], [0.9, 0.1]])

    assert written == 2
    assert vectors.count() == 2
    assert vectors.count_for_document(DOC_A) == 2
    assert vectors.contents_for_document(DOC_A) == ("第一段", "第二段")


def test_repeated_upsert_is_idempotent(vectors: ChunkVectorStore) -> None:
    chunks = make_document_chunks(DOC_A, "第一段", "第二段")

    vectors.upsert(chunks, [[1.0, 0.0], [0.9, 0.1]])
    vectors.upsert(chunks, [[1.0, 0.0], [0.9, 0.1]])

    assert vectors.count() == 2
    assert vectors.ids_for_document(DOC_A) == tuple(chunk.chunk_id for chunk in chunks)


def test_upsert_replaces_content_for_the_same_chunk_id(vectors: ChunkVectorStore) -> None:
    vectors.upsert(make_document_chunks(DOC_A, "旧内容"), [[1.0, 0.0]])

    vectors.upsert(make_document_chunks(DOC_A, "新内容"), [[1.0, 0.0]])

    assert vectors.count() == 1
    assert vectors.contents_for_document(DOC_A) == ("新内容",)


def test_delete_document_only_touches_its_own_vectors(vectors: ChunkVectorStore) -> None:
    vectors.upsert(make_document_chunks(DOC_A, "甲", "乙"), [[1.0, 0.0], [0.8, 0.2]])
    vectors.upsert(make_document_chunks(DOC_B, "丙"), [[0.0, 1.0]])

    removed = vectors.delete_document(DOC_A)

    assert removed == 2
    assert vectors.count_for_document(DOC_A) == 0
    assert vectors.count_for_document(DOC_B) == 1
    assert vectors.count() == 1


def test_delete_missing_document_removes_nothing(vectors: ChunkVectorStore) -> None:
    assert vectors.delete_document("doc-missing") == 0
    assert vectors.delete_ids([]) == 0


def test_query_returns_the_closest_chunk_first(vectors: ChunkVectorStore) -> None:
    chunks = make_document_chunks(DOC_A, "甲的正文", "乙的正文")
    vectors.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])

    hits = vectors.query([0.95, 0.05], top_k=2)

    assert hits[0].chunk.chunk_id == chunks[0].chunk_id
    assert hits[0].chunk.content == "甲的正文"
    assert hits[0].chunk.document_id == DOC_A
    assert hits[0].chunk.heading is None
    assert hits[0].chunk.page is None
    assert hits[0].distance <= hits[1].distance


def test_query_can_filter_by_source(vectors: ChunkVectorStore) -> None:
    vectors.upsert(make_document_chunks(DOC_A, "甲", source="a.md"), [[1.0, 0.0]])
    vectors.upsert(make_document_chunks(DOC_B, "乙", source="b.md"), [[1.0, 0.0]])

    hits = vectors.query([1.0, 0.0], top_k=5, source="b.md")

    assert [hit.chunk.source for hit in hits] == ["b.md"]


def test_mismatched_input_lengths_are_rejected(vectors: ChunkVectorStore) -> None:
    chunks = make_document_chunks(DOC_A, "只有一段")

    with pytest.raises(ValueError, match="same length"):
        vectors.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])


def test_upserting_nothing_is_a_no_op(vectors: ChunkVectorStore) -> None:
    assert vectors.upsert((), ()) == 0
