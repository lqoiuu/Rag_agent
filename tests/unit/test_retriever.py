"""Top-K retrieval and the confidence decision it supports."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from ingestion_test_support import make_document_chunks, make_vector_store

from rag_agent.domain.retrieval import DEFAULT_THRESHOLD, DEFAULT_TOP_K
from rag_agent.providers.fake import FakeEmbeddingModel
from rag_agent.retrieval import Retriever, RetrieverConfig
from rag_agent.vectorstore import ChunkVectorStore

DIMENSION = 16


def seed(
    vectors: ChunkVectorStore,
    model: FakeEmbeddingModel,
    document_id: str,
    contents: Sequence[str],
    *,
    source: str = "raw/manual.md",
) -> None:
    """Write chunks together with the deterministic vectors for their text."""

    chunks = make_document_chunks(document_id, *contents, source=source)
    vectors.upsert(chunks, model.embed([chunk.content for chunk in chunks]).vectors)


def make_retriever(*contents: str, config: RetrieverConfig | None = None) -> Retriever:
    vectors = make_vector_store()
    model = FakeEmbeddingModel(dimension=DIMENSION)
    if contents:
        seed(vectors, model, "doc-test", contents)
    return Retriever(vectors=vectors, embedding_model=model, config=config)


def test_exact_content_query_hits_that_chunk_first() -> None:
    retriever = make_retriever("主刷卡住时先断电检查", "充电座指示灯不亮怎么办", "滤网多久清洗一次")

    result = retriever.search("主刷卡住时先断电检查")

    assert result.hits[0].chunk.content == "主刷卡住时先断电检查"
    assert result.hits[0].rank == 1
    assert result.hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert result.is_confident is True
    assert "reaches the threshold" in result.reason


def test_hits_are_ranked_by_descending_score() -> None:
    retriever = make_retriever("主刷卡住", "滤网清洗", "充电故障")

    result = retriever.search("主刷卡住")

    scores = [hit.score for hit in result.hits]
    assert scores == sorted(scores, reverse=True)
    assert [hit.rank for hit in result.hits] == list(range(1, len(scores) + 1))


def test_top_k_override_limits_the_number_of_hits() -> None:
    retriever = make_retriever(*[f"第{index}段内容" for index in range(6)])

    result = retriever.search("第1段内容", top_k=2)

    assert len(result.hits) == 2
    assert result.top_k == 2


def test_configured_top_k_is_used_by_default() -> None:
    retriever = make_retriever(
        *[f"第{index}段内容" for index in range(6)],
        config=RetrieverConfig(top_k=3, threshold=0.0),
    )

    result = retriever.search("第1段内容")

    assert len(result.hits) == 3
    assert result.top_k == 3


def test_unrelated_query_stays_below_a_high_threshold() -> None:
    retriever = make_retriever("主刷卡住", "滤网清洗周期")

    result = retriever.search("今天天气怎么样", threshold=0.99)

    assert result.is_confident is False
    assert result.best_score is not None
    assert result.best_score < 0.99
    assert "below the threshold" in result.reason


def test_threshold_override_can_accept_any_hit() -> None:
    retriever = make_retriever("主刷卡住")

    result = retriever.search("完全无关的问题", threshold=-1.0)

    assert result.hits
    assert result.is_confident is True


def test_empty_index_is_not_confident() -> None:
    retriever = make_retriever()

    result = retriever.search("任何问题", threshold=-1.0)

    assert result.hits == ()
    assert result.best_score is None
    assert result.is_confident is False
    assert "no indexed chunk" in result.reason


def test_source_filter_restricts_the_hits() -> None:
    vectors = make_vector_store()
    model = FakeEmbeddingModel(dimension=DIMENSION)
    seed(vectors, model, "doc-a", ["甲的正文"], source="a.md")
    seed(vectors, model, "doc-b", ["乙的正文"], source="b.md")
    retriever = Retriever(vectors=vectors, embedding_model=model)

    result = retriever.search("乙的正文", source="b.md")

    assert result.hits
    assert {hit.chunk.source for hit in result.hits} == {"b.md"}


def test_chunks_carry_page_heading_and_range() -> None:
    retriever = make_retriever("主刷卡住时的处理步骤")

    result = retriever.search("主刷卡住时的处理步骤")

    chunk = result.hits[0].chunk
    assert chunk.document_id == "doc-test"
    assert chunk.source == "raw/manual.md"
    assert chunk.char_range[1] > chunk.char_range[0]
    assert chunk.chunk_id.startswith("doc-test-c")


def test_blank_query_is_rejected() -> None:
    retriever = make_retriever("内容")

    with pytest.raises(ValueError, match="must not be blank"):
        retriever.search("   ")


@pytest.mark.parametrize(
    "config",
    [
        {"top_k": 0},
        {"top_k": -3},
        {"threshold": 1.5},
        {"threshold": -2.0},
    ],
)
def test_invalid_config_is_rejected(config: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RetrieverConfig(**config)  # type: ignore[arg-type]


def test_invalid_call_overrides_are_rejected() -> None:
    retriever = make_retriever("内容")

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("内容", top_k=0)

    with pytest.raises(ValueError, match="threshold"):
        retriever.search("内容", threshold=5.0)


def test_defaults_are_the_documented_starting_points() -> None:
    config = RetrieverConfig()

    assert config.top_k == DEFAULT_TOP_K
    assert config.threshold == DEFAULT_THRESHOLD


def test_margin_reports_the_gap_between_the_two_best_hits() -> None:
    retriever = make_retriever("第一段内容", "第二段内容", "第三段内容")

    result = retriever.search("第一段内容", top_k=3)

    assert len(result.hits) >= 2
    assert result.margin == round(result.hits[0].score - result.hits[1].score, 6)
    assert result.as_dict()["margin"] == result.margin


def test_margin_is_none_with_a_single_hit() -> None:
    retriever = make_retriever("唯一分片")

    result = retriever.search("唯一分片", top_k=1)

    assert result.margin is None
    assert result.as_dict()["margin"] is None


def test_result_serialises_for_reports() -> None:
    retriever = make_retriever("主刷卡住")

    payload = retriever.search("主刷卡住").as_dict()

    assert payload["query"] == "主刷卡住"
    assert payload["confident"] is True
    assert isinstance(payload["hits"], list)
    hit = payload["hits"][0]  # type: ignore[index]
    assert set(hit) >= {"rank", "score", "chunk_id", "source", "page", "heading", "content"}
