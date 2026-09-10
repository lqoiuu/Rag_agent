"""Domain model behaviour for documents, pages and chunks."""

from __future__ import annotations

import dataclasses

import pytest

from rag_agent.domain.documents import (
    DocumentChunk,
    DocumentPage,
    DocumentStatus,
    FileType,
    SourceDocument,
    build_chunk_id,
    build_document_id,
    checksum_of,
    file_type_for_suffix,
    version_of,
)
from rag_agent.domain.errors import UnsupportedFileTypeError


def make_document(**overrides: object) -> SourceDocument:
    payload = b"first page\n\nsecond page"
    checksum = checksum_of(payload)
    values: dict[str, object] = {
        "document_id": build_document_id("raw/manual.md"),
        "source": "raw/manual.md",
        "file_type": FileType.MARKDOWN,
        "checksum": checksum,
        "version": version_of(checksum),
        "pages": (
            DocumentPage(page_number=1, text="first page"),
            DocumentPage(page_number=2, text="second page"),
        ),
        "size_bytes": len(payload),
        "encoding": "utf-8-sig",
    }
    values.update(overrides)
    return SourceDocument(**values)  # type: ignore[arg-type]


def test_file_type_is_derived_from_the_suffix() -> None:
    assert file_type_for_suffix(".txt") is FileType.TXT
    assert file_type_for_suffix(".MD") is FileType.MARKDOWN
    assert file_type_for_suffix(".markdown") is FileType.MARKDOWN
    assert file_type_for_suffix(".pdf") is FileType.PDF


def test_unsupported_suffix_lists_the_supported_types() -> None:
    with pytest.raises(UnsupportedFileTypeError) as excinfo:
        file_type_for_suffix(".docx", source="手册.docx")

    assert excinfo.value.code == "unsupported_file_type"
    assert excinfo.value.source == "手册.docx"
    assert ".pdf" in str(excinfo.value)


def test_document_id_depends_only_on_the_source_label() -> None:
    assert build_document_id("raw/manual.md") == build_document_id("raw/manual.md")
    assert build_document_id("raw/manual.md") != build_document_id("raw/faq.md")
    assert build_document_id("raw/manual.md").startswith("doc-")


def test_content_change_keeps_identity_and_changes_version() -> None:
    first = checksum_of(b"version one")
    second = checksum_of(b"version two")

    assert first != second
    assert version_of(first) == first[:12]
    assert version_of(first) != version_of(second)


def test_document_id_is_independent_of_content() -> None:
    document = make_document()
    edited_checksum = checksum_of(b"new content")
    edited = make_document(checksum=edited_checksum, version=version_of(edited_checksum))

    assert document.document_id == edited.document_id
    assert document.version != edited.version


def test_document_exposes_pages_as_normalized_text() -> None:
    document = make_document()

    assert document.page_count == 2
    assert document.text == "first page\n\nsecond page"
    assert document.status is DocumentStatus.PARSED
    assert document.duplicate_of is None


def test_duplicate_status_is_derived_from_duplicate_of() -> None:
    document = make_document(duplicate_of="raw/manual.md")

    assert document.status is DocumentStatus.DUPLICATE
    assert document.duplicate_of == "raw/manual.md"


def test_documents_are_immutable() -> None:
    document = make_document()

    with pytest.raises(dataclasses.FrozenInstanceError):
        document.source = "raw/other.md"  # type: ignore[misc]


def test_chunk_id_is_deterministic_and_ordered() -> None:
    assert build_chunk_id("doc-abc", 1) == "doc-abc-c0001"
    assert build_chunk_id("doc-abc", 12) == "doc-abc-c0012"
    assert build_chunk_id("doc-abc", 1) != build_chunk_id("doc-abc", 2)


def test_chunk_defaults_are_explicit() -> None:
    chunk = DocumentChunk(
        chunk_id=build_chunk_id("doc-abc", 1),
        document_id="doc-abc",
        source="raw/manual.md",
        content="充不进电时先检查充电座。",
        index=1,
        char_range=(0, 13),
    )

    assert chunk.page is None
    assert chunk.heading is None
    assert chunk.char_range == (0, 13)
