"""Chroma-backed storage for chunk vectors.

Embeddings are always computed by the project's own provider and passed in, so
Chroma never loads its default embedding model and the provider abstraction
stays the single source of vectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb

from rag_agent.domain.documents import DocumentChunk

DEFAULT_COLLECTION = "rag_agent_chunks"


class ChunkVectorStore:
    """Persist chunk vectors keyed by the stable chunk ID."""

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        *,
        collection_name: str = DEFAULT_COLLECTION,
        client: Any | None = None,
    ) -> None:
        if client is None:
            if persist_dir is None:
                raise ValueError("persist_dir is required when no client is given")
            directory = Path(persist_dir)
            directory.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(directory))

        self._client: Any = client
        self._collection: Any = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection_name(self) -> str:
        return str(self._collection.name)

    def upsert(self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]]) -> int:
        """Write chunks idempotently; re-writing the same chunk ID updates it."""

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return 0

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[list(vector) for vector in vectors],
            documents=[chunk.content for chunk in chunks],
            metadatas=[_metadata(chunk) for chunk in chunks],
        )
        return len(chunks)

    def ids_for_document(self, document_id: str) -> tuple[str, ...]:
        result = self._collection.get(where={"document_id": document_id}, include=[])
        return tuple(str(item) for item in result["ids"])

    def count_for_document(self, document_id: str) -> int:
        return len(self.ids_for_document(document_id))

    def contents_for_document(self, document_id: str) -> tuple[str, ...]:
        result = self._collection.get(where={"document_id": document_id}, include=["documents"])
        documents = result["documents"]
        if documents is None:
            return ()
        return tuple(str(item) for item in documents)

    def delete_ids(self, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        self._collection.delete(ids=list(ids))
        return len(ids)

    def delete_document(self, document_id: str) -> int:
        """Remove every vector of one document; returns how many were removed."""

        return self.delete_ids(self.ids_for_document(document_id))

    def count(self) -> int:
        return int(self._collection.count())

    def query(
        self, vector: Sequence[float], *, top_k: int = 5
    ) -> tuple[tuple[str, float, dict[str, Any]], ...]:
        """Return ``(chunk_id, distance, metadata)`` triples for a query vector."""

        result = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=top_k,
            include=["distances", "metadatas"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        metadatas = result["metadatas"][0]
        return tuple(
            (str(chunk_id), float(distance), dict(metadata))
            for chunk_id, distance, metadata in zip(ids, distances, metadatas, strict=True)
        )


def _metadata(chunk: DocumentChunk) -> dict[str, Any]:
    """Chroma metadata values must be scalars, so absent fields become defaults."""

    return {
        "document_id": chunk.document_id,
        "source": chunk.source,
        "chunk_index": chunk.index,
        "char_start": chunk.char_range[0],
        "char_end": chunk.char_range[1],
        "page": chunk.page if chunk.page is not None else 0,
        "heading": chunk.heading if chunk.heading is not None else "",
    }
