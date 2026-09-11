"""Evaluation runners, driven by deterministic offline models."""

from __future__ import annotations

import json

import pytest
from ingestion_test_support import make_vector_store

from rag_agent.domain.documents import DocumentChunk, build_chunk_id
from rag_agent.evaluation.dataset import EvalCase, parse_cases
from rag_agent.evaluation.runner import run_answer_evaluation, run_retrieval_evaluation
from rag_agent.providers.fake import FakeChatModel, FakeEmbeddingModel
from rag_agent.retrieval import Retriever
from rag_agent.vectorstore import ChunkVectorStore

DIMENSION = 16
PAGE_27_TEXT = "产品型号 DBX23 主机额定输入 20V 2A 充电时间约 6.5h"
PAGE_2_TEXT = "禁止在潮湿或有积水的地面上使用产品"
PAGE_3_TEXT = "地图丢失时可将主机搬至全能基站前方尝试恢复"


def make_chunk(document_id: str, content: str, *, page: int, index: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=build_chunk_id(document_id, index),
        document_id=document_id,
        source="manual.pdf",
        content=content,
        index=index,
        char_range=(0, len(content)),
        page=page,
    )


def make_retriever() -> Retriever:
    vectors: ChunkVectorStore = make_vector_store()
    model = FakeEmbeddingModel(dimension=DIMENSION)
    chunks = (
        make_chunk("doc-1", PAGE_2_TEXT, page=2, index=1),
        make_chunk("doc-1", PAGE_3_TEXT, page=3, index=2),
        make_chunk("doc-1", PAGE_27_TEXT, page=27, index=3),
    )
    vectors.upsert(chunks, model.embed([chunk.content for chunk in chunks]).vectors)
    return Retriever(vectors=vectors, embedding_model=model)


def make_cases() -> tuple[EvalCase, ...]:
    return parse_cases(
        [
            json.dumps(
                {
                    "id": "a",
                    "question": PAGE_27_TEXT,
                    "category": "产品参数",
                    "answerable": True,
                    "expected_pages": [27],
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "id": "b",
                    "question": PAGE_2_TEXT,
                    "category": "安全规范",
                    "answerable": True,
                    "expected_pages": [2],
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "id": "c",
                    "question": "这台机器人明年会涨价吗",
                    "category": "范围外",
                    "answerable": False,
                    "expected_pages": [],
                },
                ensure_ascii=False,
            ),
        ]
    )


def reply(citation: int, *, status: str = "answered", answer: str = "型号是 DBX23。[1]") -> str:
    return json.dumps(
        {"status": status, "answer": answer, "citations": [citation], "reason": ""},
        ensure_ascii=False,
    )


def test_retrieval_run_matches_expected_pages() -> None:
    report = run_retrieval_evaluation(
        make_cases(),
        retriever=make_retriever(),
        top_k=3,
        conditions={"dataset": "memory", "top_k": 3},
    )

    assert report.mode == "retrieval"
    assert report.summary.total == 3
    assert report.summary.recall_at_k == 1.0
    assert report.summary.mrr == 1.0
    assert report.conditions["dataset"] == "memory"
    assert report.categories == {"产品参数": 1, "安全规范": 1, "范围外": 1}
    assert [case.case_id for case in report.cases] == ["a", "b", "c"]
    assert report.cases[2].retrieval is not None
    assert report.cases[2].retrieval.recall_at_k is False


def test_retrieval_run_reports_a_miss() -> None:
    cases = parse_cases(
        [
            json.dumps(
                {
                    "id": "x",
                    "question": PAGE_27_TEXT,
                    "category": "产品参数",
                    "answerable": True,
                    "expected_pages": [99],
                },
                ensure_ascii=False,
            )
        ]
    )

    report = run_retrieval_evaluation(cases, retriever=make_retriever(), top_k=3)

    assert report.summary.recall_at_k == 0.0
    assert report.summary.mrr == 0.0


def test_answer_run_measures_citations_and_faithfulness() -> None:
    cases = make_cases()[:1]
    model = FakeChatModel([reply(1, answer=PAGE_27_TEXT)])

    report = run_answer_evaluation(
        cases,
        retriever=make_retriever(),
        chat_model=model,
        top_k=3,
    )

    assert report.mode == "answer"
    assert report.summary.answer_rate == 1.0
    assert report.summary.citation_correct_rate == 1.0
    assert report.summary.citation_repair_attempts == 0
    assert report.summary.citation_repairs == 0
    assert report.summary.mean_faithfulness == 1.0
    assert report.cases[0].citation_pages == (27,)
    assert report.cases[0].citation_repair_attempted is False
    assert report.cases[0].citation_repaired is False


def test_answer_run_counts_a_wrong_citation() -> None:
    cases = make_cases()[:1]
    model = FakeChatModel([reply(2, answer="地图丢失时可以搬至基站前方恢复")])

    report = run_answer_evaluation(cases, retriever=make_retriever(), chat_model=model, top_k=3)

    assert report.summary.citation_correct_rate == 0.0
    assert report.cases[0].citation_pages == (2,)


def test_answer_run_records_refusals_for_unanswerable_cases() -> None:
    cases = make_cases()[2:]
    model = FakeChatModel(
        [
            json.dumps(
                {
                    "status": "insufficient",
                    "answer": "",
                    "citations": [],
                    "reason": "资料未涉及价格与定价政策。",
                },
                ensure_ascii=False,
            )
        ]
    )

    report = run_answer_evaluation(cases, retriever=make_retriever(), chat_model=model, top_k=3)

    assert report.summary.refusal_accuracy == 1.0
    assert report.summary.decision_accuracy == 1.0
    assert report.cases[0].refusal_cause == "model_insufficient"
    assert report.summary.refusal_breakdown == {"model_insufficient": 1}


def test_answer_run_without_evidence_refuses_without_calling_the_model() -> None:
    cases = make_cases()
    model = FakeChatModel([reply(1)])
    empty_retriever = Retriever(vectors=make_vector_store(), embedding_model=FakeEmbeddingModel())

    report = run_answer_evaluation(cases, retriever=empty_retriever, chat_model=model, top_k=3)

    assert model.call_count == 0
    assert report.summary.answer_rate == 0.0
    assert report.summary.refusal_accuracy == 1.0
    assert report.summary.refusal_breakdown == {"no_hits": 3}


def test_report_serialises_every_case() -> None:
    report = run_retrieval_evaluation(make_cases(), retriever=make_retriever(), top_k=3)

    payload = report.as_dict()

    assert payload["mode"] == "retrieval"
    assert len(payload["cases"]) == 3  # type: ignore[arg-type]
    assert "summary" in payload


@pytest.mark.parametrize("mode", ["retrieval"])
def test_report_markdown_mentions_conditions_and_metrics(mode: str) -> None:
    from rag_agent.evaluation.report import to_markdown

    report = run_retrieval_evaluation(
        make_cases(), retriever=make_retriever(), top_k=3, conditions={"top_k": 3}
    )

    markdown = to_markdown(report)

    assert "# RAG 评测报告" in markdown
    assert "Recall@K" in markdown
    assert "top_k：3" in markdown
    assert "| a |" in markdown
