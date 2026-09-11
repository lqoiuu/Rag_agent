"""Render an evaluation report as Markdown or JSON."""

from __future__ import annotations

import json

from rag_agent.evaluation.metrics import CaseOutcome
from rag_agent.evaluation.runner import EvaluationReport

_METRIC_LABELS: tuple[tuple[str, str], ...] = (
    ("recall_at_k", "Recall@K"),
    ("mrr", "MRR"),
    ("answer_rate", "回答率"),
    ("citation_correct_rate", "引用正确率"),
    ("refusal_accuracy", "拒答正确率"),
    ("decision_accuracy", "决策正确率"),
    ("mean_faithfulness", "忠实度（代理指标）"),
    ("mean_best_score", "平均最高相似度"),
    ("mean_margin", "平均前两名分差"),
)


def to_json(report: EvaluationReport) -> str:
    """Machine-readable report, including every per-case number."""

    return json.dumps(report.as_dict(), ensure_ascii=False, indent=2)


def to_markdown(report: EvaluationReport) -> str:
    """Human-readable report with the conditions needed to reproduce it."""

    summary = report.summary
    lines: list[str] = [
        f"# RAG 评测报告（{summary.mode} 模式）",
        "",
        "## 实验条件",
        "",
    ]
    for key, value in report.conditions.items():
        lines.append(f"- {key}：{value}")
    lines.extend(
        [
            "",
            "## 数据集",
            "",
            f"- 用例总数：{summary.total}"
            f"（可回答 {summary.answerable}，不可回答 {summary.unanswerable}）",
        ]
    )
    for category, count in report.categories.items():
        lines.append(f"- {category}：{count}")
    lines.extend(["", "## 指标", "", "| 指标 | 数值 |", "|---|---|"])
    for key, label in _METRIC_LABELS:
        value = getattr(summary, key)
        lines.append(f"| {label} | {'—' if value is None else value} |")
    if summary.refusal_breakdown:
        lines.extend(["", "## 拒答原因分布", "", "| 原因 | 次数 |", "|---|---|"])
        for cause, count in summary.refusal_breakdown.items():
            lines.append(f"| {cause} | {count} |")
    lines.extend(
        [
            "",
            "## 逐条结果",
            "",
            "| 用例 | 类别 | 可回答 | 首个命中 | 命中 | 回答 | 引用页码 | 引用正确 | 忠实度 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for case in report.cases:
        rank = (
            "—"
            if case.retrieval is None or case.retrieval.first_match_rank is None
            else case.retrieval.first_match_rank
        )
        hit = "—" if case.retrieval is None else ("是" if case.retrieval.recall_at_k else "否")
        answered = "—" if case.answered is None else ("是" if case.answered else "否")
        pages = ",".join(str(page) for page in case.citation_pages) or "—"
        correct = (
            "—" if case.citation_correct is None else ("是" if case.citation_correct else "否")
        )
        score = "—" if case.faithfulness is None else case.faithfulness
        able = "是" if case.answerable else "否"
        lines.append(
            f"| {case.case_id} | {case.category} | {able} | {rank} | {hit} | {answered} | "
            f"{pages} | {correct} | {score} |"
        )

    problems = [
        case
        for case in report.cases
        if (case.answerable and case.answered is False)
        or case.citation_correct is False
        or (case.answerable and case.retrieval is not None and not case.retrieval.recall_at_k)
    ]
    if problems:
        lines.extend(
            [
                "",
                "## 需要关注的用例",
                "",
                "| 用例 | 问题 | 说明 |",
                "|---|---|---|",
            ]
        )
        for case in problems:
            lines.append(f"| {case.case_id} | {_problem_label(case)} | {_problem_detail(case)} |")

    lines.append("")
    return "\n".join(lines)


def _problem_label(case: CaseOutcome) -> str:
    labels: list[str] = []
    if case.answerable and case.retrieval is not None and not case.retrieval.recall_at_k:
        labels.append("检索未命中")
    if case.answerable and case.answered is False:
        labels.append(f"该答未答（{case.refusal_cause}）")
    if case.citation_correct is False:
        labels.append("引用不在预期页")
    return "、".join(labels) or "—"


def _problem_detail(case: CaseOutcome) -> str:
    detail = case.answer_excerpt or case.raw_excerpt
    if not detail:
        return "—"
    compact = " ".join(detail.split())
    return compact.replace("|", "/")[:160]
