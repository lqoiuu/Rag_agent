"""Chunk length distribution used as experiment evidence."""

from __future__ import annotations

import pytest
from ingestion_test_support import make_chunks

from rag_agent.ingestion.stats import summarize_chunks


def test_summary_reports_count_and_central_tendency() -> None:
    stats = summarize_chunks(make_chunks(100, 200, 300))

    assert stats.chunk_count == 3
    assert stats.total_chars == 600
    assert stats.min_chars == 100
    assert stats.max_chars == 300
    assert stats.median_chars == 200.0
    assert stats.mean_chars == 200.0


def test_p90_uses_nearest_rank() -> None:
    stats = summarize_chunks(make_chunks(10, 20, 30, 40, 50, 60, 70, 80, 90, 100))

    assert stats.p90_chars == 90
    assert stats.median_chars == 55.0


def test_histogram_buckets_group_lengths() -> None:
    stats = summarize_chunks(make_chunks(50, 250, 450), bucket_size=200)

    assert stats.histogram == (
        ("0-199", 1),
        ("200-399", 1),
        ("400-599", 1),
    )


def test_histogram_grows_to_cover_the_longest_chunk() -> None:
    stats = summarize_chunks(make_chunks(120, 520), bucket_size=200)

    assert [label for label, _count in stats.histogram] == ["0-199", "200-399", "400-599"]
    assert sum(count for _label, count in stats.histogram) == 2


def test_empty_chunk_set_is_summarised_as_zeros() -> None:
    stats = summarize_chunks(())

    assert stats.chunk_count == 0
    assert stats.total_chars == 0
    assert stats.max_chars == 0
    assert stats.median_chars == 0.0
    assert stats.histogram == ()


def test_single_chunk_is_its_own_minimum_and_maximum() -> None:
    stats = summarize_chunks(make_chunks(742))

    assert stats.chunk_count == 1
    assert stats.min_chars == stats.max_chars == stats.p90_chars == 742
    assert stats.mean_chars == 742.0


def test_non_positive_bucket_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        summarize_chunks(make_chunks(10), bucket_size=0)


def test_summary_serialises_for_reports() -> None:
    payload = summarize_chunks(make_chunks(50, 250), bucket_size=200).as_dict()

    assert payload["chunk_count"] == 2
    assert payload["total_chars"] == 300
    assert payload["histogram"] == [["0-199", 1], ["200-399", 1]]
