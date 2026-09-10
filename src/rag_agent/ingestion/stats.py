"""Chunk statistics used by the chunking experiments.

The numbers here are the evidence that stage 8 will compare across chunk sizes,
so they are plain data rather than formatted text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from rag_agent.domain.documents import DocumentChunk

DEFAULT_BUCKET_SIZE = 200


@dataclass(frozen=True, slots=True)
class ChunkStats:
    """Length distribution of one chunk set."""

    chunk_count: int
    total_chars: int
    min_chars: int
    median_chars: float
    p90_chars: int
    max_chars: int
    mean_chars: float
    histogram: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_count": self.chunk_count,
            "total_chars": self.total_chars,
            "min_chars": self.min_chars,
            "median_chars": self.median_chars,
            "p90_chars": self.p90_chars,
            "max_chars": self.max_chars,
            "mean_chars": self.mean_chars,
            "histogram": [list(item) for item in self.histogram],
        }


def summarize_chunks(
    chunks: Sequence[DocumentChunk], *, bucket_size: int = DEFAULT_BUCKET_SIZE
) -> ChunkStats:
    """Summarize chunk count and length distribution.

    Percentiles use the nearest-rank method so the reported value is always a
    real chunk length rather than an interpolated one.
    """

    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")

    lengths = sorted(len(chunk.content) for chunk in chunks)
    if not lengths:
        return ChunkStats(
            chunk_count=0,
            total_chars=0,
            min_chars=0,
            median_chars=0.0,
            p90_chars=0,
            max_chars=0,
            mean_chars=0.0,
            histogram=(),
        )

    total = sum(lengths)
    return ChunkStats(
        chunk_count=len(lengths),
        total_chars=total,
        min_chars=lengths[0],
        median_chars=float(median(lengths)),
        p90_chars=_nearest_rank(lengths, 90),
        max_chars=lengths[-1],
        mean_chars=round(total / len(lengths), 1),
        histogram=_histogram(lengths, bucket_size),
    )


def _nearest_rank(sorted_lengths: Sequence[int], percentile: int) -> int:
    position = max(1, -(-percentile * len(sorted_lengths) // 100))
    return sorted_lengths[min(position, len(sorted_lengths)) - 1]


def _histogram(sorted_lengths: Sequence[int], bucket_size: int) -> tuple[tuple[str, int], ...]:
    counts: list[tuple[str, int]] = []
    upper = bucket_size - 1
    while True:
        lower = upper - bucket_size + 1
        if lower == 0:
            count = sum(1 for length in sorted_lengths if length <= upper)
        else:
            count = sum(1 for length in sorted_lengths if lower <= length <= upper)
        counts.append((f"{lower}-{upper}", count))
        if upper >= sorted_lengths[-1]:
            break
        upper += bucket_size
    return tuple(counts)
