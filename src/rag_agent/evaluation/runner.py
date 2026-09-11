"""Evaluation runners that turn a dataset into measured outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag_agent.evaluation.dataset import EvalCase, category_counts
from rag_agent.evaluation.metrics import (
    CaseOutcome,
    EvaluationSummary,
    faithfulness,
    score_retrieval,
    summarize,
)
from rag_agent.generation.rag_answer import answer_with_context
from rag_agent.providers.base import ChatModel
from rag_agent.retrieval.retriever import Retriever


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """One evaluation run: conditions, per-case outcomes and the summary."""

    mode: str
    conditions: dict[str, Any]
    cases: tuple[CaseOutcome, ...]
    summary: EvaluationSummary
    categories: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "conditions": dict(self.conditions),
            "categories": dict(self.categories),
            "summary": self.summary.as_dict(),
            "cases": [case.as_dict() for case in self.cases],
        }


def run_retrieval_evaluation(
    cases: Sequence[EvalCase],
    *,
    retriever: Retriever,
    top_k: int | None = None,
    threshold: float | None = None,
    conditions: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Measure retrieval only: no chat model, so no answer cost."""

    outcomes: list[CaseOutcome] = []
    for case in cases:
        result = retriever.search(case.question, top_k=top_k, threshold=threshold)
        outcomes.append(
            CaseOutcome(
                case_id=case.case_id,
                category=case.category,
                answerable=case.answerable,
                retrieval=score_retrieval(result.hits, case.expected_pages),
            )
        )

    return _build_report("retrieval", cases, outcomes, conditions)


def run_answer_evaluation(
    cases: Sequence[EvalCase],
    *,
    retriever: Retriever,
    chat_model: ChatModel,
    top_k: int | None = None,
    threshold: float | None = None,
    conditions: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Measure retrieval plus grounded answering, including refusals."""

    outcomes: list[CaseOutcome] = []
    for case in cases:
        answer = answer_with_context(
            case.question,
            retriever=retriever,
            chat_model=chat_model,
            top_k=top_k,
            threshold=threshold,
        )
        citation_pages = tuple(
            citation.page for citation in answer.citations if citation.page is not None
        )
        expected = set(case.expected_pages)
        citation_correct: bool | None = None
        score: float | None = None
        if answer.answered:
            citation_correct = bool(citation_pages) and all(
                page in expected for page in citation_pages
            )
            # Compare against the full cited chunk text, not the shortened
            # excerpt, otherwise the proxy under-reports for long answers.
            cited_texts = [
                answer.hits[citation.citation_id - 1].chunk.content
                for citation in answer.citations
                if 0 < citation.citation_id <= len(answer.hits)
            ]
            score = faithfulness(answer.text, cited_texts)

        outcomes.append(
            CaseOutcome(
                case_id=case.case_id,
                category=case.category,
                answerable=case.answerable,
                question=case.question,
                retrieval=score_retrieval(answer.hits, case.expected_pages),
                answered=answer.answered,
                refusal_cause=None if answer.refusal_cause is None else str(answer.refusal_cause),
                citation_pages=citation_pages,
                citation_correct=citation_correct,
                faithfulness=score,
                latency_ms=answer.latency_ms,
                answer_excerpt=answer.text[:160],
                raw_excerpt="" if answer.answered else answer.raw_output[:200],
                citation_repair_attempted=answer.citation_repair_attempted,
                citation_repaired=answer.citation_repaired,
            )
        )

    return _build_report("answer", cases, outcomes, conditions)


def _build_report(
    mode: str,
    cases: Sequence[EvalCase],
    outcomes: Sequence[CaseOutcome],
    conditions: dict[str, Any] | None,
) -> EvaluationReport:
    return EvaluationReport(
        mode=mode,
        conditions=dict(conditions or {}),
        cases=tuple(outcomes),
        summary=summarize(outcomes, mode=mode),
        categories=category_counts(cases),
    )
