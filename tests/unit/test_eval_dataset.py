"""Evaluation dataset parsing and validation."""

from __future__ import annotations

import json

import pytest

from rag_agent.evaluation.dataset import (
    EvalCase,
    EvaluationDatasetError,
    category_counts,
    parse_cases,
)


def case_line(**overrides: object) -> str:
    payload: dict[str, object] = {
        "id": "q01",
        "question": "产品型号是什么",
        "category": "产品参数",
        "answerable": True,
        "expected_pages": [27],
        "reference_answer": "DBX23",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_valid_line_becomes_a_case() -> None:
    cases = parse_cases([case_line()])

    assert len(cases) == 1
    assert cases[0] == EvalCase(
        case_id="q01",
        question="产品型号是什么",
        category="产品参数",
        answerable=True,
        expected_pages=(27,),
        reference_answer="DBX23",
    )


def test_blank_lines_and_comments_are_ignored() -> None:
    cases = parse_cases(["", "   ", "# 这是注释", case_line()])

    assert len(cases) == 1


def test_unanswerable_case_must_not_list_pages() -> None:
    with pytest.raises(EvaluationDatasetError, match="not answerable"):
        parse_cases([case_line(answerable=False)])


def test_answerable_case_requires_a_page() -> None:
    with pytest.raises(EvaluationDatasetError, match="no expected page"):
        parse_cases([case_line(expected_pages=[])])


def test_missing_field_is_reported() -> None:
    payload = json.loads(case_line())
    del payload["category"]

    with pytest.raises(EvaluationDatasetError, match="missing fields"):
        parse_cases([json.dumps(payload, ensure_ascii=False)])


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(EvaluationDatasetError, match="duplicate case id"):
        parse_cases([case_line(), case_line()])


def test_empty_question_is_rejected() -> None:
    with pytest.raises(EvaluationDatasetError, match="empty question"):
        parse_cases([case_line(question="   ")])


def test_pages_must_be_positive_integers() -> None:
    with pytest.raises(EvaluationDatasetError, match="positive integers"):
        parse_cases([case_line(expected_pages=[0])])

    with pytest.raises(EvaluationDatasetError, match="must be a list"):
        parse_cases([case_line(expected_pages=27)])


def test_invalid_json_is_reported_with_its_line() -> None:
    with pytest.raises(EvaluationDatasetError, match="line 2 is not valid JSON"):
        parse_cases([case_line(), "{not json}"])


def test_empty_dataset_is_rejected() -> None:
    with pytest.raises(EvaluationDatasetError, match="no cases"):
        parse_cases(["# 只有注释"])


def test_category_counts_are_sorted() -> None:
    cases = parse_cases(
        [
            case_line(id="a", category="安全规范"),
            case_line(id="b", category="产品参数"),
            case_line(id="c", category="安全规范"),
        ]
    )

    assert category_counts(cases) == {"产品参数": 1, "安全规范": 2}


def test_shipped_dataset_is_valid_and_large_enough() -> None:
    from pathlib import Path

    dataset = Path("data/eval/qa_set.jsonl")
    if not dataset.is_file():  # pragma: no cover - 数据集缺失时跳过
        pytest.skip("shipped evaluation set is not available")

    cases = parse_cases(dataset.read_text(encoding="utf-8").splitlines())

    assert len(cases) >= 30
    assert sum(1 for case in cases if case.answerable) >= 25
    assert sum(1 for case in cases if not case.answerable) >= 5
