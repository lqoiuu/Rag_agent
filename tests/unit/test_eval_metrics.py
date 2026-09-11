"""Retrieval and answer metrics, verified against hand-computed values."""

from __future__ import annotations

from rag_agent.domain.documents import DocumentChunk, build_chunk_id
from rag_agent.domain.retrieval import RetrievalHit
from rag_agent.evaluation.metrics import (
    CaseOutcome,
    faithfulness,
    score_retrieval,
    summarize,
)


def hit(page: int, *, rank: int, score: float) -> RetrievalHit:
    return RetrievalHit(
        chunk=DocumentChunk(
            chunk_id=build_chunk_id("doc-1", rank),
            document_id="doc-1",
            source="manual.pdf",
            content=f"第{page}页内容",
            index=rank,
            char_range=(0, 10),
            page=page,
        ),
        score=score,
        rank=rank,
    )


def test_matching_hit_at_rank_one_scores_full_recall() -> None:
    hits = [hit(27, rank=1, score=0.6), hit(3, rank=2, score=0.5)]

    score = score_retrieval(hits, [27])

    assert score.recall_at_k is True
    assert score.first_match_rank == 1
    assert score.reciprocal_rank == 1.0
    assert score.best_score == 0.6
    assert score.margin == 0.1


def test_matching_hit_at_rank_three_uses_reciprocal_rank() -> None:
    hits = [hit(3, rank=1, score=0.6), hit(4, rank=2, score=0.5), hit(27, rank=3, score=0.4)]

    score = score_retrieval(hits, [27])

    assert score.first_match_rank == 3
    assert score.reciprocal_rank == 1 / 3
    assert score.recall_at_k is True


def test_no_matching_page_is_a_miss() -> None:
    score = score_retrieval([hit(3, rank=1, score=0.6)], [27])

    assert score.recall_at_k is False
    assert score.first_match_rank is None
    assert score.reciprocal_rank == 0.0


def test_any_expected_page_counts_as_a_match() -> None:
    score = score_retrieval([hit(4, rank=1, score=0.6)], [1, 2, 8])

    assert score.recall_at_k is False

    score = score_retrieval([hit(8, rank=1, score=0.6)], [1, 2, 8])

    assert score.recall_at_k is True


def test_empty_hits_report_no_scores() -> None:
    score = score_retrieval([], [27])

    assert score.hit_count == 0
    assert score.best_score is None
    assert score.margin is None


def test_faithfulness_is_one_for_copied_text() -> None:
    text = "先断电并清理主刷"

    assert faithfulness(text, [text + "。"]) == 1.0


def test_faithfulness_is_zero_for_unrelated_text() -> None:
    assert faithfulness("今天天气不错", ["滤网每两周清洗一次"]) == 0.0


def test_faithfulness_handles_short_and_empty_answers() -> None:
    assert faithfulness("", ["任意内容"]) == 0.0
    assert faithfulness("短", ["短"]) == 0.0


def test_retrieval_summary_aggregates_recall_and_mrr() -> None:
    outcomes = [
        CaseOutcome(
            case_id="a",
            category="产品参数",
            answerable=True,
            retrieval=score_retrieval([hit(27, rank=1, score=0.6)], [27]),
        ),
        CaseOutcome(
            case_id="b",
            category="产品参数",
            answerable=True,
            retrieval=score_retrieval([hit(3, rank=1, score=0.6)], [27]),
        ),
        CaseOutcome(case_id="c", category="范围外", answerable=False),
    ]

    summary = summarize(outcomes, mode="retrieval")

    assert summary.total == 3
    assert summary.answerable == 2
    assert summary.unanswerable == 1
    assert summary.recall_at_k == 0.5
    assert summary.mrr == 0.5
    assert summary.answer_rate is None
    assert summary.refusal_accuracy is None


def test_answer_summary_separates_citation_and_refusal_quality() -> None:
    outcomes = [
        CaseOutcome(
            case_id="a",
            category="产品参数",
            answerable=True,
            answered=True,
            citation_pages=(27,),
            citation_correct=True,
            faithfulness=0.9,
            latency_ms=1000.0,
        ),
        CaseOutcome(
            case_id="b",
            category="故障排查",
            answerable=True,
            answered=True,
            citation_pages=(3,),
            citation_correct=False,
            faithfulness=0.4,
            latency_ms=2000.0,
            citation_repair_attempted=True,
            citation_repaired=True,
        ),
        CaseOutcome(
            case_id="c",
            category="范围外",
            answerable=False,
            answered=False,
            refusal_cause="model_insufficient",
            latency_ms=3000.0,
        ),
    ]

    summary = summarize(outcomes, mode="answer")

    assert summary.answer_rate == 1.0
    assert summary.citation_correct_rate == 0.5
    assert summary.refusal_accuracy == 1.0
    assert summary.decision_accuracy == 1.0
    assert summary.mean_faithfulness == 0.65
    assert summary.latency_p50_ms == 2000.0
    assert summary.latency_p95_ms == 2900.0
    assert summary.citation_repair_attempts == 1
    assert summary.citation_repairs == 1
    assert summary.refusal_breakdown == {"model_insufficient": 1}


def test_refusing_an_answerable_question_counts_as_a_wrong_decision() -> None:
    outcomes = [
        CaseOutcome(
            case_id="a",
            category="产品参数",
            answerable=True,
            answered=False,
            refusal_cause="no_hits",
        ),
        CaseOutcome(case_id="b", category="范围外", answerable=False, answered=True),
    ]

    summary = summarize(outcomes, mode="answer")

    assert summary.decision_accuracy == 0.0
    assert summary.answer_rate == 0.0
    assert summary.refusal_accuracy == 0.0


def test_summary_serialises_for_reports() -> None:
    summary = summarize([], mode="retrieval")

    payload = summary.as_dict()

    assert payload["mode"] == "retrieval"
    assert payload["total"] == 0
    assert payload["recall_at_k"] is None
    assert payload["latency_p50_ms"] is None
    assert payload["latency_p95_ms"] is None
    assert payload["citation_repair_attempts"] is None
    assert payload["citation_repairs"] is None
    assert payload["refusal_breakdown"] == {}
