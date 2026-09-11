"""Evaluation dataset loading and validation.

The set is plain JSON Lines so it can be diffed, reviewed and extended without a
build step. Every case declares whether the knowledge base can answer it, which
is what makes refusal accuracy measurable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("id", "question", "category", "answerable", "expected_pages")


class EvaluationDatasetError(ValueError):
    """The dataset is missing, malformed or internally inconsistent."""

    code = "evaluation_dataset_error"


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One evaluation item."""

    case_id: str
    question: str
    category: str
    answerable: bool
    expected_pages: tuple[int, ...]
    reference_answer: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "question": self.question,
            "category": self.category,
            "answerable": self.answerable,
            "expected_pages": list(self.expected_pages),
            "reference_answer": self.reference_answer,
        }


def load_cases(path: Path | str) -> tuple[EvalCase, ...]:
    """Load and validate a JSON Lines evaluation set."""

    dataset = Path(path)
    if not dataset.is_file():
        raise EvaluationDatasetError(f"evaluation set does not exist: {dataset}")
    return parse_cases(dataset.read_text(encoding="utf-8").splitlines())


def parse_cases(lines: Iterable[str]) -> tuple[EvalCase, ...]:
    """Parse JSON Lines into cases, rejecting anything inconsistent."""

    cases: list[EvalCase] = []
    seen_ids: set[str] = set()

    for number, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise EvaluationDatasetError(f"line {number} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise EvaluationDatasetError(f"line {number} must be a JSON object")

        missing = [field for field in REQUIRED_FIELDS if field not in payload]
        if missing:
            raise EvaluationDatasetError(f"line {number} is missing fields: {', '.join(missing)}")

        case = _to_case(payload, number)
        if case.case_id in seen_ids:
            raise EvaluationDatasetError(f"duplicate case id {case.case_id!r} on line {number}")
        seen_ids.add(case.case_id)
        cases.append(case)

    if not cases:
        raise EvaluationDatasetError("evaluation set contains no cases")
    return tuple(cases)


def _to_case(payload: dict[str, Any], number: int) -> EvalCase:
    question = str(payload["question"]).strip()
    if not question:
        raise EvaluationDatasetError(f"line {number} has an empty question")

    answerable = bool(payload["answerable"])
    pages = tuple(_as_page(value, number) for value in _as_pages(payload["expected_pages"], number))
    if answerable and not pages:
        raise EvaluationDatasetError(f"line {number} is answerable but lists no expected page")
    if not answerable and pages:
        raise EvaluationDatasetError(f"line {number} is not answerable but lists expected pages")

    return EvalCase(
        case_id=str(payload["id"]),
        question=question,
        category=str(payload["category"]),
        answerable=answerable,
        expected_pages=pages,
        reference_answer=str(payload.get("reference_answer", "")),
    )


def _as_pages(value: object, number: int) -> Iterator[object]:
    if not isinstance(value, list):
        raise EvaluationDatasetError(f"line {number} field expected_pages must be a list")
    return iter(value)


def _as_page(value: object, number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvaluationDatasetError(f"line {number} expected_pages must contain positive integers")
    return value


def category_counts(cases: Iterable[EvalCase]) -> dict[str, int]:
    """Count cases per category, for the report header."""

    counts: dict[str, int] = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    return dict(sorted(counts.items()))
