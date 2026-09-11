"""Indexing pipeline: idempotency, replacement, removal and failure handling."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest
from ingestion_test_support import make_document, make_document_chunks, make_vector_store

from rag_agent.domain.documents import SourceDocument
from rag_agent.ingestion.pipeline import (
    FAILED,
    INDEXED,
    REMOVED,
    UNCHANGED,
    embed_chunks,
    index_document,
    ingest_path,
    remove_document,
)
from rag_agent.ingestion.splitters import ChunkingConfig, split_document
from rag_agent.providers.base import EmbeddingResponse, ModelTimeoutError
from rag_agent.providers.fake import FakeEmbeddingModel
from rag_agent.storage import MetadataStore
from rag_agent.vectorstore import ChunkVectorStore

SMALL_CHUNKS = ChunkingConfig(chunk_size=120, chunk_overlap=0)


class RecordingEmbeddingModel:
    """Deterministic embeddings that record the batches they were asked for."""

    def __init__(self, *, dimension: int = 8, errors: Sequence[Exception] = ()) -> None:
        self._inner = FakeEmbeddingModel(dimension=dimension)
        self._errors = list(errors)
        self.batches: list[tuple[str, ...]] = []

    @property
    def model_name(self) -> str:
        return "fake-embedding"

    def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        if self._errors:
            raise self._errors.pop(0)
        batch = tuple(texts)
        self.batches.append(batch)
        return self._inner.embed(batch)

    @property
    def embedded(self) -> list[str]:
        return [text for batch in self.batches for text in batch]


def long_document() -> SourceDocument:
    return make_document(["# 标题\n" + "这是一段较长的正文内容。" * 40])


def short_document() -> SourceDocument:
    return make_document(["# 标题\n短正文。"])


@pytest.fixture
def stores() -> Iterator[tuple[MetadataStore, ChunkVectorStore]]:
    with MetadataStore(":memory:") as store:
        yield store, make_vector_store()


def test_indexing_writes_vectors_and_metadata(
    stores: tuple[MetadataStore, ChunkVectorStore],
) -> None:
    store, vectors = stores
    model = RecordingEmbeddingModel()
    document = long_document()

    result = index_document(
        document, store=store, vectors=vectors, embedding_model=model, config=SMALL_CHUNKS
    )

    assert result.status == INDEXED
    assert result.chunk_count == len(split_document(document, SMALL_CHUNKS))
    assert result.chunk_count > 1
    assert result.vector_count == result.chunk_count
    assert vectors.count_for_document(document.document_id) == result.chunk_count

    stored = store.get_document(document.source)
    assert stored is not None
    assert stored.checksum == document.checksum
    assert stored.chunk_count == result.chunk_count
    assert stored.embedding_model == "fake-embedding"

    job = store.latest_job(document.source)
    assert job is not None
    assert job.status == "succeeded"
    assert job.chunk_count == result.chunk_count


def test_reindexing_an_unchanged_document_skips_embedding(
    stores: tuple[MetadataStore, ChunkVectorStore],
) -> None:
    store, vectors = stores
    model = RecordingEmbeddingModel()
    document = long_document()

    first = index_document(
        document, store=store, vectors=vectors, embedding_model=model, config=SMALL_CHUNKS
    )
    batches_after_first = len(model.batches)

    second = index_document(
        document, store=store, vectors=vectors, embedding_model=model, config=SMALL_CHUNKS
    )

    assert second.status == UNCHANGED
    assert second.chunk_count == first.chunk_count
    assert len(model.batches) == batches_after_first
    assert vectors.count_for_document(document.document_id) == first.chunk_count

    job = store.latest_job(document.source)
    assert job is not None
    assert job.status == "skipped"


def test_force_reindexes_an_unchanged_document(
    stores: tuple[MetadataStore, ChunkVectorStore],
) -> None:
    store, vectors = stores
    model = RecordingEmbeddingModel()
    document = long_document()
    index_document(
        document, store=store, vectors=vectors, embedding_model=model, config=SMALL_CHUNKS
    )
    batches_after_first = len(model.batches)

    forced = index_document(
        document,
        store=store,
        vectors=vectors,
        embedding_model=model,
        config=SMALL_CHUNKS,
        force=True,
    )

    assert forced.status == INDEXED
    assert len(model.batches) > batches_after_first


def test_changed_content_replaces_the_previous_chunks(
    stores: tuple[MetadataStore, ChunkVectorStore],
) -> None:
    store, vectors = stores
    model = RecordingEmbeddingModel()

    first = index_document(
        long_document(), store=store, vectors=vectors, embedding_model=model, config=SMALL_CHUNKS
    )
    second = index_document(
        short_document(), store=store, vectors=vectors, embedding_model=model, config=SMALL_CHUNKS
    )

    document_id = short_document().document_id
    assert first.chunk_count > second.chunk_count
    assert second.status == INDEXED
    assert vectors.count_for_document(document_id) == second.chunk_count

    stored = store.get_document(short_document().source)
    assert stored is not None
    assert stored.checksum == short_document().checksum
    assert stored.chunk_count == second.chunk_count
    assert len(store.list_versions(document_id)) == 2


def test_embedding_failure_is_recorded_and_leaves_no_document_row(
    stores: tuple[MetadataStore, ChunkVectorStore],
) -> None:
    store, vectors = stores
    model = RecordingEmbeddingModel(errors=[ModelTimeoutError("scripted timeout")])

    result = index_document(
        long_document(),
        store=store,
        vectors=vectors,
        embedding_model=model,
        config=SMALL_CHUNKS,
    )

    assert result.status == FAILED
    assert result.error_code == "timeout"
    assert store.get_document("raw/manual.md") is None
    assert vectors.count() == 0

    job = store.latest_job("raw/manual.md")
    assert job is not None
    assert job.status == "failed"
    assert job.error_code == "timeout"


def test_failure_keeps_the_previous_version_indexed(
    stores: tuple[MetadataStore, ChunkVectorStore],
) -> None:
    store, vectors = stores
    working = RecordingEmbeddingModel()
    first = index_document(
        long_document(),
        store=store,
        vectors=vectors,
        embedding_model=working,
        config=SMALL_CHUNKS,
    )
    failing = RecordingEmbeddingModel(errors=[ModelTimeoutError("scripted timeout")])

    result = index_document(
        short_document(),
        store=store,
        vectors=vectors,
        embedding_model=failing,
        config=SMALL_CHUNKS,
    )

    document_id = long_document().document_id
    stored = store.get_document("raw/manual.md")
    assert result.status == FAILED
    assert stored is not None
    assert stored.checksum == long_document().checksum
    assert vectors.count_for_document(document_id) == first.chunk_count


def test_remove_document_clears_both_stores(
    stores: tuple[MetadataStore, ChunkVectorStore],
) -> None:
    store, vectors = stores
    model = RecordingEmbeddingModel()
    result = index_document(
        long_document(), store=store, vectors=vectors, embedding_model=model, config=SMALL_CHUNKS
    )

    removed = remove_document("raw/manual.md", store=store, vectors=vectors)

    assert removed.status == REMOVED
    assert removed.vector_count == result.chunk_count
    assert vectors.count_for_document(long_document().document_id) == 0
    assert store.get_document("raw/manual.md") is None

    again = remove_document("raw/manual.md", store=store, vectors=vectors)
    assert again.status == FAILED
    assert again.error_code == "document_not_found"


def test_ingest_path_reports_a_missing_file(
    stores: tuple[MetadataStore, ChunkVectorStore],
) -> None:
    store, vectors = stores
    model = RecordingEmbeddingModel()

    result = ingest_path(
        "definitely-missing.txt", store=store, vectors=vectors, embedding_model=model
    )

    assert result.status == FAILED
    assert result.error_code == "document_not_found"
    assert model.batches == []

    job = store.latest_job("definitely-missing.txt")
    assert job is not None
    assert job.status == "failed"
    assert job.error_code == "document_not_found"


def test_embed_chunks_respects_the_provider_batch_size() -> None:
    chunks = make_document_chunks("doc-batch", *[f"内容{index}" for index in range(23)])
    model = RecordingEmbeddingModel()

    vectors = embed_chunks(model, chunks)

    assert [len(batch) for batch in model.batches] == [10, 10, 3]
    assert len(vectors) == 23
    assert model.embedded[0] == "内容0"
    assert model.embedded[-1] == "内容22"


def toc_entry(title: str, page: int, dots: int = 70) -> str:
    """Build one table-of-contents line with dot leaders."""

    return f"{title} {'.' * dots}{page}"


TABLE_OF_CONTENTS = "\n".join(
    (
        "3",
        "目录",
        toc_entry("1. 产品组成", 4),
        toc_entry(" 1.1 包装内容物", 4),
        toc_entry(" 1.2 部件名称", 4),
        toc_entry("2. 产品使用", 8),
        toc_entry("     2.1 注意事项", 8),
    )
)


def test_index_like_chunks_are_dropped_before_embedding(
    stores: tuple[MetadataStore, ChunkVectorStore],
) -> None:
    store, vectors = stores
    model = RecordingEmbeddingModel()
    document = make_document([TABLE_OF_CONTENTS + "\n\n" + "正常的说明文字。" * 40])

    result = index_document(
        document, store=store, vectors=vectors, embedding_model=model, config=SMALL_CHUNKS
    )

    total = len(split_document(document, SMALL_CHUNKS))
    assert result.dropped_chunks >= 1
    assert result.chunk_count == total - result.dropped_chunks
    assert result.vector_count == result.chunk_count
    assert result.as_dict()["dropped_chunks"] == result.dropped_chunks
    assert "目录" not in " ".join(model.embedded)


def test_index_like_chunks_can_be_kept(
    stores: tuple[MetadataStore, ChunkVectorStore],
) -> None:
    store, vectors = stores
    model = RecordingEmbeddingModel()
    document = make_document([TABLE_OF_CONTENTS + "\n\n" + "正常的说明文字。" * 40])

    result = index_document(
        document,
        store=store,
        vectors=vectors,
        embedding_model=model,
        config=SMALL_CHUNKS,
        drop_index_like_chunks=False,
    )

    total = len(split_document(document, SMALL_CHUNKS))
    assert result.dropped_chunks == 0
    assert result.chunk_count == total
    assert "目录" in " ".join(model.embedded)
