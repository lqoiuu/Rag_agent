"""Metrics for retrieval and answer quality.

Everything here is a pure function over already-computed results, so the numbers
in a report can be recomputed and unit-tested without any model call.

Two deliberate limitations are recorded rather than hidden:

- Retrieval is scored at *page* granularity, because that is the level the
  evaluation set can be labelled reliably without human chunk-level annotation.
- Faithfulness is a *proxy*: how much of the answer's character n-grams occur in
  the cited chunks. It can be fooled by copying, and it cannot detect a wrong
  conclusion drawn from the right text, so it is reported next to — never
  instead of — citation correctness and refusal accuracy.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rag_agent.domain.retrieval import RetrievalHit

NGRAM_SIZE = 4


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """Retrieval quality of one question."""

    first_match_rank: int | None
    recall_at_k: bool
    reciprocal_rank: float
    best_score: float | None
    margin: float | None
    hit_count: int


def score_retrieval(hits: Sequence[RetrievalHit], expected_pages: Sequence[int]) -> RetrievalScore:
    """Score one retrieval result against the expected pages."""

    expected = set(expected_pages)
    first_match: int | None = None
    for hit in hits:
        if hit.chunk.page is not None and hit.chunk.page in expected:
            first_match = hit.rank
            break

    best_score = hits[0].score if hits else None
    margin = None if len(hits) < 2 else round(hits[0].score - hits[1].score, 6)
    return RetrievalScore(
        first_match_rank=first_match,
        recall_at_k=first_match is not None,
        reciprocal_rank=0.0 if first_match is None else 1.0 / first_match,
        best_score=best_score,
        margin=margin,
        hit_count=len(hits),
    )


def faithfulness(answer: str, cited_texts: Sequence[str], *, ngram: int = NGRAM_SIZE) -> float:
    """Proxy score: fraction of the answer's n-grams found in the cited text.

    Both sides are compared with whitespace removed, because Chinese text mixes
    full-width characters with spaced Latin words and a whitespace-sensitive
    comparison would under-report support for no good reason.
    """

    compact_answer = _compact(answer)
    answer_grams = _ngrams(compact_answer, ngram)
    if not answer_grams:
        return 0.0
    cited = "".join(_compact(text) for text in cited_texts)
    supported = sum(1 for gram in answer_grams if gram in cited)
    return round(supported / len(answer_grams), 4)


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """Everything measured for one case, in either mode."""

    case_id: str
    category: str
    answerable: bool
    question: str = ""
    retrieval: RetrievalScore | None = None
    answered: bool | None = None
    refusal_cause: str | None = None
    citation_pages: tuple[int, ...] = ()
    citation_correct: bool | None = None
    faithfulness: float | None = None
    latency_ms: float | None = None
    answer_excerpt: str = ""
    raw_excerpt: str = ""
    citation_repair_attempted: bool = False
    citation_repaired: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "category": self.category,
            "answerable": self.answerable,
            "question": self.question,
            "first_match_rank": None if self.retrieval is None else self.retrieval.first_match_rank,
            "recall_at_k": None if self.retrieval is None else self.retrieval.recall_at_k,
            "reciprocal_rank": (
                None if self.retrieval is None else round(self.retrieval.reciprocal_rank, 4)
            ),
            "best_score": None if self.retrieval is None else self.retrieval.best_score,
            "margin": None if self.retrieval is None else self.retrieval.margin,
            "answered": self.answered,
            "refusal_cause": self.refusal_cause,
            "citation_pages": list(self.citation_pages),
            "citation_correct": self.citation_correct,
            "faithfulness": self.faithfulness,
            "latency_ms": self.latency_ms,
            "answer_excerpt": self.answer_excerpt,
            "raw_excerpt": self.raw_excerpt,
            "citation_repair_attempted": self.citation_repair_attempted,
            "citation_repaired": self.citation_repaired,
        }


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Aggregated metrics of one evaluation run."""

    mode: str
    total: int
    answerable: int
    unanswerable: int
    recall_at_k: float | None
    mrr: float | None
    answer_rate: float | None
    citation_correct_rate: float | None
    refusal_accuracy: float | None
    decision_accuracy: float | None
    mean_faithfulness: float | None
    mean_best_score: float | None
    mean_margin: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    citation_repair_attempts: int | None
    citation_repairs: int | None
    refusal_breakdown: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "total": self.total,
            "answerable": self.answerable,
            "unanswerable": self.unanswerable,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "answer_rate": self.answer_rate,
            "citation_correct_rate": self.citation_correct_rate,
            "refusal_accuracy": self.refusal_accuracy,
            "decision_accuracy": self.decision_accuracy,
            "mean_faithfulness": self.mean_faithfulness,
            "mean_best_score": self.mean_best_score,
            "mean_margin": self.mean_margin,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "citation_repair_attempts": self.citation_repair_attempts,
            "citation_repairs": self.citation_repairs,
            "refusal_breakdown": self.refusal_breakdown,
        }


def summarize(outcomes: Sequence[CaseOutcome], *, mode: str) -> EvaluationSummary:
    """Aggregate per-case outcomes into one summary."""

    answerable = [item for item in outcomes if item.answerable]
    unanswerable = [item for item in outcomes if not item.answerable]
    scored = [item.retrieval for item in answerable if item.retrieval is not None]

    refused_unanswerable = [item for item in unanswerable if item.answered is False]

    breakdown: dict[str, int] = {}
    for item in outcomes:
        if item.refusal_cause:
            breakdown[item.refusal_cause] = breakdown.get(item.refusal_cause, 0) + 1
    if mode == "retrieval":
        return EvaluationSummary(
            mode=mode,
            total=len(outcomes),
            answerable=len(answerable),
            unanswerable=len(unanswerable),
            recall_at_k=_mean_bool([item.recall_at_k for item in scored]),
            mrr=_mean([item.reciprocal_rank for item in scored]),
            answer_rate=None,
            citation_correct_rate=None,
            refusal_accuracy=None,
            decision_accuracy=None,
            mean_faithfulness=None,
            mean_best_score=_mean(
                [item.best_score for item in scored if item.best_score is not None]
            ),
            mean_margin=_mean([item.margin for item in scored if item.margin is not None]),
            latency_p50_ms=None,
            latency_p95_ms=None,
            citation_repair_attempts=None,
            citation_repairs=None,
            refusal_breakdown=breakdown,
        )

    answered_answerable = [item for item in answerable if item.answered is True]
    correct_citations = [item for item in answered_answerable if item.citation_correct is True]
    decisions = sum(1 for item in outcomes if _decided_correctly(item))
    latencies = [item.latency_ms for item in outcomes if item.latency_ms is not None]

    return EvaluationSummary(
        mode=mode,
        total=len(outcomes),
        answerable=len(answerable),
        unanswerable=len(unanswerable),
        recall_at_k=_mean_bool([item.recall_at_k for item in scored]),
        mrr=_mean([item.reciprocal_rank for item in scored]),
        answer_rate=_ratio(len(answered_answerable), len(answerable)),
        citation_correct_rate=_ratio(len(correct_citations), len(answered_answerable)),
        refusal_accuracy=_ratio(len(refused_unanswerable), len(unanswerable)),
        decision_accuracy=_ratio(decisions, len(outcomes)),
        mean_faithfulness=_mean(
            [item.faithfulness for item in answered_answerable if item.faithfulness is not None]
        ),
        mean_best_score=_mean([item.best_score for item in scored if item.best_score is not None]),
        mean_margin=_mean([item.margin for item in scored if item.margin is not None]),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        citation_repair_attempts=sum(1 for item in outcomes if item.citation_repair_attempted),
        citation_repairs=sum(1 for item in outcomes if item.citation_repaired),
        refusal_breakdown=breakdown,
    )


def _decided_correctly(outcome: CaseOutcome) -> bool:
    if outcome.answered is None:
        return False
    return outcome.answered if outcome.answerable else not outcome.answered


def _compact(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def _ngrams(compact: str, size: int) -> set[str]:
    if len(compact) < size:
        return set()
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return round(sum(items) / len(items), 4)


def _mean_bool(values: Iterable[bool]) -> float | None:
    items = list(values)
    if not items:
        return None
    return round(sum(1 for item in items if item) / len(items), 4)


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile from measured values."""

    items = sorted(values)
    if not items:
        return None
    position = (len(items) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(items) - 1)
    fraction = position - lower
    return round(items[lower] + (items[upper] - items[lower]) * fraction, 1)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)
