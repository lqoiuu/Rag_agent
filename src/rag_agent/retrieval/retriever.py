"""Top-K semantic retrieval with an explicit confidence decision."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rag_agent.domain.retrieval import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    RetrievalHit,
    RetrievalResult,
)
from rag_agent.providers.base import EmbeddingModel
from rag_agent.vectorstore.chroma import ChunkVectorStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    """Retrieval parameters that can be adjusted per experiment."""

    top_k: int = DEFAULT_TOP_K
    threshold: float = DEFAULT_THRESHOLD

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not -1.0 <= self.threshold <= 1.0:
            raise ValueError(
                "threshold must be within [-1, 1] because scores are cosine similarities"
            )


class Retriever:
    """Embed a query and return the closest indexed chunks.

    The vector store reports a cosine distance, so the score kept on each hit is
    ``1 - distance``. Whether that score is good enough is a separate decision,
    expressed by the threshold and exposed as ``RetrievalResult.is_confident``.
    """

    def __init__(
        self,
        *,
        vectors: ChunkVectorStore,
        embedding_model: EmbeddingModel,
        config: RetrieverConfig | None = None,
    ) -> None:
        self._vectors = vectors
        self._embedding_model = embedding_model
        self._config = config if config is not None else RetrieverConfig()

    @property
    def config(self) -> RetrieverConfig:
        return self._config

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        source: str | None = None,
    ) -> RetrievalResult:
        """Retrieve the closest chunks for one query.

        ``top_k``, ``threshold`` and ``source`` override the configured defaults
        for a single call, which is what the debug command uses.
        """

        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query must not be blank")

        resolved_top_k = self._config.top_k if top_k is None else top_k
        resolved_threshold = self._config.threshold if threshold is None else threshold
        if resolved_top_k <= 0:
            raise ValueError("top_k must be positive")
        if not -1.0 <= resolved_threshold <= 1.0:
            raise ValueError("threshold must be within [-1, 1]")

        response = self._embedding_model.embed([cleaned])
        matches = self._vectors.query(
            response.vectors[0], top_k=resolved_top_k, source=source or None
        )
        hits = tuple(
            RetrievalHit(chunk=match.chunk, score=round(1.0 - match.distance, 6), rank=rank)
            for rank, match in enumerate(matches, start=1)
        )
        result = RetrievalResult(
            query=cleaned,
            hits=hits,
            threshold=resolved_threshold,
            top_k=resolved_top_k,
        )
        LOGGER.info(
            "retrieval query_chars=%d hits=%d best=%s confident=%s source=%s",
            len(cleaned),
            len(hits),
            result.best_score,
            result.is_confident,
            source or "-",
        )
        return result
