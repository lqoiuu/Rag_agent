"""Heading-aware chunking: traceability, boundaries and the default splitter."""

from __future__ import annotations

import pytest
from ingestion_test_support import fixed_pieces, make_document

from rag_agent.domain.documents import build_chunk_id
from rag_agent.domain.errors import ChunkingError, MissingDependencyError
from rag_agent.ingestion import splitters
from rag_agent.ingestion.splitters import (
    ChunkingConfig,
    build_langchain_splitter,
    split_document,
    split_into_sections,
)


def test_sections_keep_the_heading_hierarchy() -> None:
    page = "# 充电问题\n正文一\n## 充不进电\n正文二\n### 检查项\n正文三"

    sections = split_into_sections(page)

    assert [section.heading for section in sections] == [
        "充电问题",
        "充电问题 > 充不进电",
        "充电问题 > 充不进电 > 检查项",
    ]
    assert [section.heading_level for section in sections] == [1, 2, 3]
    assert sections[0].text.startswith("# 充电问题")
    assert sections[2].text.startswith("### 检查项")


def test_text_before_the_first_heading_is_untitled() -> None:
    sections = split_into_sections("前言内容\n# 标题\n正文")

    assert sections[0].heading is None
    assert sections[0].heading_level == 0
    assert sections[0].text.strip() == "前言内容"
    assert sections[1].heading == "标题"


def test_same_level_heading_replaces_the_previous_sibling() -> None:
    sections = split_into_sections("# A\n## B\n# C")

    assert [section.heading for section in sections] == ["A", "A > B", "C"]


def test_blank_pages_produce_no_sections() -> None:
    assert split_into_sections("") == ()
    assert split_into_sections("   \n\n  ") == ()


def test_chunks_are_numbered_and_bound_to_their_heading() -> None:
    document = make_document(["# 标题\n第一段内容"])

    chunks = split_document(document, split_text=fixed_pieces("第一段内容"))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == build_chunk_id(document.document_id, 1)
    assert chunk.index == 1
    assert chunk.heading == "标题"
    assert chunk.page == 1
    start, end = chunk.char_range
    assert document.text[start:end] == chunk.content


def test_every_chunk_can_be_located_in_the_whole_document() -> None:
    document = make_document(["第一页内容", "第二页内容"])

    chunks = split_document(document, split_text=lambda text: [text])

    assert [chunk.page for chunk in chunks] == [1, 2]
    assert [chunk.index for chunk in chunks] == [1, 2]
    for chunk in chunks:
        start, end = chunk.char_range
        assert document.text[start:end] == chunk.content


def test_overlapping_pieces_are_located_correctly() -> None:
    document = make_document(["abcdefghij"])

    chunks = split_document(
        document,
        ChunkingConfig(chunk_size=6, chunk_overlap=4),
        split_text=fixed_pieces("abcdef", "efghij"),
    )

    assert [chunk.content for chunk in chunks] == ["abcdef", "efghij"]
    assert [chunk.char_range for chunk in chunks] == [(0, 6), (4, 10)]


def test_repeated_text_is_located_at_the_right_offset() -> None:
    document = make_document(["XXXXXXXXXXXX"])

    chunks = split_document(
        document,
        ChunkingConfig(chunk_size=4, chunk_overlap=0),
        split_text=fixed_pieces("XXXX", "XXXX", "XXXX"),
    )

    assert [chunk.char_range for chunk in chunks] == [(0, 4), (4, 8), (8, 12)]


def test_empty_and_blank_pieces_are_skipped() -> None:
    document = make_document(["正文内容"])

    chunks = split_document(document, split_text=fixed_pieces("正文内容", "", "   ", "\n"))

    assert [chunk.content for chunk in chunks] == ["正文内容"]
    assert len(chunks) == 1


def test_text_outside_the_source_is_rejected() -> None:
    document = make_document(["正文内容"])

    with pytest.raises(ChunkingError) as excinfo:
        split_document(document, split_text=fixed_pieces("这段文字不在原文里"))

    assert excinfo.value.code == "chunking_error"
    assert excinfo.value.source == "raw/manual.md"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": -10},
        {"chunk_overlap": -1},
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_size": 100, "chunk_overlap": 200},
        {"separators": ()},
    ],
)
def test_invalid_configs_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)  # type: ignore[arg-type]


def test_config_serialises_for_reports() -> None:
    payload = ChunkingConfig(chunk_size=400, chunk_overlap=40).as_dict()

    assert payload["chunk_size"] == 400
    assert payload["chunk_overlap"] == 40
    assert isinstance(payload["separators"], list)


def test_missing_splitter_dependency_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(splitters.importlib, "import_module", boom)

    with pytest.raises(MissingDependencyError) as excinfo:
        build_langchain_splitter(ChunkingConfig(), source="raw/manual.md")

    assert excinfo.value.code == "missing_dependency"
    assert excinfo.value.source == "raw/manual.md"


def test_langchain_splitter_respects_chunk_size_and_headings() -> None:
    pytest.importorskip("langchain_text_splitters")
    document = make_document(["# 标题\n" + "这是一段用于测试分片的正文内容。" * 100])
    config = ChunkingConfig(chunk_size=200, chunk_overlap=20)

    chunks = split_document(document, config)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= config.chunk_size for chunk in chunks)
    assert all(chunk.heading == "标题" for chunk in chunks)
    assert all(chunk.page == 1 for chunk in chunks)
    for chunk in chunks:
        start, end = chunk.char_range
        assert document.text[start:end] == chunk.content


def test_langchain_splitter_bounds_text_without_separators() -> None:
    pytest.importorskip("langchain_text_splitters")
    document = make_document(["X" * 1500])
    config = ChunkingConfig(chunk_size=300, chunk_overlap=0)

    chunks = split_document(document, config)

    assert len(chunks) >= 5
    assert max(len(chunk.content) for chunk in chunks) <= config.chunk_size
    for chunk in chunks:
        start, end = chunk.char_range
        assert document.text[start:end] == chunk.content
