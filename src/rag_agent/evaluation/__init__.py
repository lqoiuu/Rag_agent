"""Offline evaluation: dataset, metrics, runner and reports.

Retrieval quality and answer quality are measured separately on purpose: they
fail for different reasons, and mixing them hides which half regressed.
"""

from rag_agent.evaluation.dataset import (
    EvalCase,
    EvaluationDatasetError,
    load_cases,
    parse_cases,
)
from rag_agent.evaluation.metrics import (
    CaseOutcome,
    EvaluationSummary,
    faithfulness,
    score_retrieval,
    summarize,
)
from rag_agent.evaluation.report import to_json, to_markdown
from rag_agent.evaluation.runner import (
    EvaluationReport,
    run_answer_evaluation,
    run_retrieval_evaluation,
)

__all__ = [
    "CaseOutcome",
    "EvalCase",
    "EvaluationDatasetError",
    "EvaluationReport",
    "EvaluationSummary",
    "faithfulness",
    "load_cases",
    "parse_cases",
    "run_answer_evaluation",
    "run_retrieval_evaluation",
    "score_retrieval",
    "summarize",
    "to_json",
    "to_markdown",
]
