"""Indexing pipeline: load, split, embed and persist one document.

Two properties matter more than speed here:

- **Idempotency** — chunk IDs are derived from the document ID and chunk order,
  so re-indexing the same content rewrites the same vectors instead of adding
  duplicates.
- **Consistency** — new vectors are written first and stale vectors of the same
  document are removed afterwards, so a shorter document never leaves orphans
  behind and a failure never empties an existing index.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from rag_agent.domain.documents import DocumentChunk, SourceDocument
from rag_agent.domain.errors import IngestionError
from rag_agent.ingestion.loaders import iter_supported_files, load_document, source_name
from rag_agent.ingestion.splitters import ChunkingConfig, split_document
from rag_agent.providers.base import EmbeddingModel, ModelError
from rag_agent.storage.sqlite import IngestionJob, MetadataStore, StoredDocument, utc_now
from rag_agent.vectorstore.chroma import ChunkVectorStore

LOGGER = logging.getLogger(__name__)

EMBEDDING_BATCH_SIZE = 10
RAW_DIRECTORY_NAME = "raw"

INDEXED = "indexed"
UNCHANGED = "unchanged"
REMOVED = "removed"
FAILED = "failed"

JOB_SUCCEEDED = "succeeded"
JOB_SKIPPED = "skipped"
JOB_FAILED = "failed"


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of one document, whether it was indexed, skipped or rejected."""

    source: str
    status: str
    document_id: str | None = None
    chunk_count: int = 0
    vector_count: int = 0
    error_code: str | None = None
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        return self.status != FAILED

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "status": self.status,
            "document_id": self.document_id,
            "chunk_count": self.chunk_count,
            "vector_count": self.vector_count,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


def raw_directory(data_dir: Path) -> Path:
    """Return the directory that holds the source documents."""

    return data_dir / RAW_DIRECTORY_NAME


def embed_chunks(
    model: EmbeddingModel, chunks: Sequence[DocumentChunk]
) -> tuple[tuple[float, ...], ...]:
    """Embed chunks in provider-sized batches, keeping input order."""

    vectors: list[tuple[float, ...]] = []
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        response = model.embed([chunk.content for chunk in batch])
        vectors.extend(tuple(float(value) for value in vector) for vector in response.vectors)
    return tuple(vectors)


def index_document(
    document: SourceDocument,
    *,
    store: MetadataStore,
    vectors: ChunkVectorStore,
    embedding_model: EmbeddingModel,
    config: ChunkingConfig | None = None,
    force: bool = False,
) -> IngestResult:
    """Index one loaded document, skipping work that is already up to date."""

    job_id = f"job-{uuid4().hex[:12]}"
    started_at = utc_now()
    model_name = embedding_model.model_name
    stored = store.get_document(document.source)

    if not force and stored is not None and _is_up_to_date(stored, document, model_name, vectors):
        store.record_job(
            IngestionJob(
                job_id=job_id,
                source=document.source,
                status=JOB_SKIPPED,
                started_at=started_at,
                finished_at=utc_now(),
                document_id=document.document_id,
                chunk_count=stored.chunk_count,
            )
        )
        return IngestResult(
            source=document.source,
            status=UNCHANGED,
            document_id=document.document_id,
            chunk_count=stored.chunk_count,
            vector_count=stored.chunk_count,
        )

    try:
        chunks = split_document(document, config)
        if not chunks:
            raise IngestionError("document produced no chunks", source=document.source)
        written = vectors.upsert(chunks, embed_chunks(embedding_model, chunks))
        stale = [
            chunk_id
            for chunk_id in vectors.ids_for_document(document.document_id)
            if chunk_id not in {chunk.chunk_id for chunk in chunks}
        ]
        vectors.delete_ids(stale)
    except (IngestionError, ModelError, ValueError) as exc:
        code = str(getattr(exc, "code", "indexing_error"))
        store.record_job(
            IngestionJob(
                job_id=job_id,
                source=document.source,
                status=JOB_FAILED,
                started_at=started_at,
                finished_at=utc_now(),
                document_id=document.document_id,
                error_code=code,
                error_message=str(exc),
            )
        )
        LOGGER.warning("indexing failed source=%s code=%s", document.source, code)
        return IngestResult(
            source=document.source,
            status=FAILED,
            document_id=document.document_id,
            error_code=code,
            error_message=str(exc),
        )

    store.save_document(
        StoredDocument(
            document_id=document.document_id,
            source=document.source,
            file_type=document.file_type.value,
            checksum=document.checksum,
            version=document.version,
            size_bytes=document.size_bytes,
            page_count=document.page_count,
            chunk_count=len(chunks),
            embedding_model=model_name,
            updated_at=utc_now(),
        )
    )
    store.record_job(
        IngestionJob(
            job_id=job_id,
            source=document.source,
            status=JOB_SUCCEEDED,
            started_at=started_at,
            finished_at=utc_now(),
            document_id=document.document_id,
            chunk_count=len(chunks),
        )
    )
    LOGGER.info(
        "indexed source=%s chunks=%d written=%d replacement=%s",
        document.source,
        len(chunks),
        written,
        stored is not None,
    )
    return IngestResult(
        source=document.source,
        status=INDEXED,
        document_id=document.document_id,
        chunk_count=len(chunks),
        vector_count=vectors.count_for_document(document.document_id),
    )


def ingest_path(
    path: Path | str,
    *,
    store: MetadataStore,
    vectors: ChunkVectorStore,
    embedding_model: EmbeddingModel,
    root: Path | None = None,
    config: ChunkingConfig | None = None,
    force: bool = False,
) -> IngestResult:
    """Load one file and index it, recording a failed job when loading fails."""

    try:
        document = load_document(path, root=root)
    except IngestionError as exc:
        source = exc.source or str(path)
        store.record_job(
            IngestionJob(
                job_id=f"job-{uuid4().hex[:12]}",
                source=source,
                status=JOB_FAILED,
                started_at=utc_now(),
                finished_at=utc_now(),
                error_code=exc.code,
                error_message=str(exc),
            )
        )
        LOGGER.warning("load failed source=%s code=%s", source, exc.code)
        return IngestResult(
            source=source, status=FAILED, error_code=exc.code, error_message=str(exc)
        )

    return index_document(
        document,
        store=store,
        vectors=vectors,
        embedding_model=embedding_model,
        config=config,
        force=force,
    )


def ingest_directory(
    directory: Path | str,
    *,
    store: MetadataStore,
    vectors: ChunkVectorStore,
    embedding_model: EmbeddingModel,
    config: ChunkingConfig | None = None,
    force: bool = False,
) -> tuple[IngestResult, ...]:
    """Index every supported file under ``directory``, isolating failures."""

    root = Path(directory)
    return tuple(
        ingest_path(
            path,
            store=store,
            vectors=vectors,
            embedding_model=embedding_model,
            root=root,
            config=config,
            force=force,
        )
        for path in iter_supported_files(root)
    )


def sync_index(
    directory: Path | str,
    *,
    store: MetadataStore,
    vectors: ChunkVectorStore,
    embedding_model: EmbeddingModel,
    config: ChunkingConfig | None = None,
) -> tuple[IngestResult, ...]:
    """Rebuild every file under ``directory`` and drop documents that vanished."""

    root = Path(directory)
    results = list(
        ingest_directory(
            root,
            store=store,
            vectors=vectors,
            embedding_model=embedding_model,
            config=config,
            force=True,
        )
    )
    for stored in store.list_documents():
        if (root / stored.source).is_file():
            continue
        removed = vectors.delete_document(stored.document_id)
        store.delete_document(stored.source)
        results.append(
            IngestResult(
                source=stored.source,
                status=REMOVED,
                document_id=stored.document_id,
                chunk_count=stored.chunk_count,
                vector_count=removed,
            )
        )
    return tuple(results)


def remove_document(
    source: str, *, store: MetadataStore, vectors: ChunkVectorStore
) -> IngestResult:
    """Delete one indexed document from both stores."""

    stored = store.get_document(source)
    if stored is None:
        return IngestResult(
            source=source,
            status=FAILED,
            error_code="document_not_found",
            error_message=f"no indexed document with source {source!r}",
        )

    removed = vectors.delete_document(stored.document_id)
    store.delete_document(source)
    LOGGER.info("removed source=%s vectors=%d", source, removed)
    return IngestResult(
        source=source,
        status=REMOVED,
        document_id=stored.document_id,
        chunk_count=stored.chunk_count,
        vector_count=removed,
    )


def resolve_source_root(path: Path, raw_dir: Path) -> Path | None:
    """Return the root used for source labels when the file lives under it."""

    resolved = path.resolve()
    try:
        resolved.relative_to(raw_dir.resolve())
    except ValueError:
        return None
    return raw_dir


def source_label(path: Path, raw_dir: Path) -> str:
    """Label a path the same way the loader would."""

    return source_name(path, root=resolve_source_root(path, raw_dir))


def _is_up_to_date(
    stored: StoredDocument,
    document: SourceDocument,
    model_name: str,
    vectors: ChunkVectorStore,
) -> bool:
    if stored.checksum != document.checksum or stored.embedding_model != model_name:
        return False
    return stored.chunk_count == vectors.count_for_document(document.document_id)
