"""Shared helpers for the ingestion test modules."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

import chromadb

from rag_agent.domain.documents import (
    DocumentChunk,
    DocumentPage,
    FileType,
    SourceDocument,
    build_chunk_id,
    build_document_id,
    checksum_of,
    version_of,
)
from rag_agent.ingestion.splitters import TextSplitter
from rag_agent.vectorstore import ChunkVectorStore


def make_vector_store() -> ChunkVectorStore:
    """Return an isolated in-memory vector store.

    Every store gets its own collection because one Chroma client per process
    shares the same in-memory system, and a collection is bound to the vector
    dimension it was first created with.
    """

    return ChunkVectorStore(
        client=chromadb.EphemeralClient(),
        collection_name=f"test-chunks-{uuid4().hex[:8]}",
    )


def make_document(
    pages: Sequence[str],
    *,
    source: str = "raw/manual.md",
    file_type: FileType = FileType.MARKDOWN,
) -> SourceDocument:
    """Build a source document without touching the file system."""

    raw = "\n\n".join(pages).encode("utf-8")
    checksum = checksum_of(raw)
    return SourceDocument(
        document_id=build_document_id(source),
        source=source,
        file_type=file_type,
        checksum=checksum,
        version=version_of(checksum),
        pages=tuple(
            DocumentPage(page_number=index, text=text) for index, text in enumerate(pages, start=1)
        ),
        size_bytes=len(raw),
    )


def fixed_pieces(*values: str) -> TextSplitter:
    """A deterministic splitter that always returns the given pieces."""

    return lambda _text: list(values)


def make_chunks(*lengths: int) -> tuple[DocumentChunk, ...]:
    """Build chunks whose only interesting property is their length."""

    return tuple(
        DocumentChunk(
            chunk_id=build_chunk_id("doc-test", index),
            document_id="doc-test",
            source="raw/manual.md",
            content="x" * length,
            index=index,
            char_range=(0, length),
        )
        for index, length in enumerate(lengths, start=1)
    )


def make_document_chunks(
    document_id: str, *contents: str, source: str = "raw/manual.md"
) -> tuple[DocumentChunk, ...]:
    """Build chunks that belong to one document, in order."""

    return tuple(
        DocumentChunk(
            chunk_id=build_chunk_id(document_id, index),
            document_id=document_id,
            source=source,
            content=content,
            index=index,
            char_range=(0, len(content)),
        )
        for index, content in enumerate(contents, start=1)
    )
