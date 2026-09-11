"""Retrieval results and the confidence decision they support.

The score is a cosine similarity (``1 - cosine distance``) reported by the
vector store, so it lies in ``[-1, 1]``. The threshold is therefore a value in
the same range, and the default below is an explicit starting point that stage 8
has to calibrate against a retrieval evaluation set rather than a quality claim.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_agent.domain.documents import DocumentChunk

DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.35


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """One retrieved chunk together with its similarity and rank."""

    chunk: DocumentChunk
    score: float
    rank: int

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "source": self.chunk.source,
            "page": self.chunk.page,
            "heading": self.chunk.heading,
            "char_range": list(self.chunk.char_range),
            "content": self.chunk.content,
        }


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Outcome of one query, including whether the evidence is trustworthy."""

    query: str
    hits: tuple[RetrievalHit, ...]
    threshold: float
    top_k: int

    @property
    def best_score(self) -> float | None:
        return self.hits[0].score if self.hits else None

    @property
    def margin(self) -> float | None:
        """Gap between the best and second best score, or None when unavailable.

        Recorded so stage 8 can test whether a relative signal separates
        answerable questions better than the absolute threshold, which stage 6
        measured to be unreliable.
        """

        if len(self.hits) < 2:
            return None
        return round(self.hits[0].score - self.hits[1].score, 6)

    @property
    def is_confident(self) -> bool:
        """True only when a hit exists and its score reaches the threshold.

        This is a coarse floor used to avoid pointless model calls. Stage 6
        measured that it cannot separate answerable from unanswerable questions,
        so it is deliberately *not* the refusal mechanism.
        """

        best = self.best_score
        return best is not None and best >= self.threshold

    @property
    def reason(self) -> str:
        """Explain the confidence decision instead of only asserting it."""

        if not self.hits:
            return "no indexed chunk matched the query"
        best = self.hits[0]
        if self.is_confident:
            return (
                f"best score {best.score:.4f} from {best.chunk.source} "
                f"reaches the threshold {self.threshold:.4f}"
            )
        return (
            f"best score {best.score:.4f} from {best.chunk.source} stays below "
            f"the threshold {self.threshold:.4f}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "threshold": self.threshold,
            "top_k": self.top_k,
            "confident": self.is_confident,
            "best_score": None if self.best_score is None else round(self.best_score, 4),
            "margin": self.margin,
            "reason": self.reason,
            "hits": [hit.as_dict() for hit in self.hits],
        }
