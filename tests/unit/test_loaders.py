"""Loader behaviour for text files, batches, failures and duplicates."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_agent.domain.documents import DocumentStatus, FileType
from rag_agent.domain.errors import (
    DocumentNotFoundError,
    EmptyDocumentError,
    EncodingDetectionError,
    UnsupportedFileTypeError,
)
from rag_agent.ingestion.loaders import (
    iter_supported_files,
    load_document,
    load_documents,
    source_name,
)


def write_file(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_bytes(text.encode(encoding))
    return path


def test_text_document_is_loaded_with_a_stable_identity(tmp_path: Path) -> None:
    source_file = write_file(
        tmp_path / "manual.txt", "第一步：检查充电座。\n\n\n第二步：检查电源。"
    )

    document = load_document(source_file, root=tmp_path)
    again = load_document(source_file, root=tmp_path)

    assert document.source == "manual.txt"
    assert document.file_type is FileType.TXT
    assert document.page_count == 1
    assert document.pages[0].page_number == 1
    assert document.pages[0].text == "第一步：检查充电座。\n\n第二步：检查电源。"
    assert document.document_id == again.document_id
    assert document.checksum == again.checksum
    assert document.version == document.checksum[:12]
    assert document.size_bytes == source_file.stat().st_size
    assert document.encoding == "utf-8"


def test_markdown_headings_are_preserved(tmp_path: Path) -> None:
    source_file = write_file(tmp_path / "guide.md", "# 充电问题\n\n充不进电时先检查充电座。\n")

    document = load_document(source_file, root=tmp_path)

    assert document.file_type is FileType.MARKDOWN
    assert document.pages[0].text.startswith("# 充电问题")


def test_legacy_encoded_file_is_decoded(tmp_path: Path) -> None:
    source_file = write_file(tmp_path / "legacy.txt", "旧编码资料", encoding="gb18030")

    document = load_document(source_file, root=tmp_path)

    assert document.pages[0].text == "旧编码资料"
    assert document.encoding == "gb18030"


def test_source_is_relative_to_the_given_root(tmp_path: Path) -> None:
    nested = tmp_path / "raw" / "manual"
    nested.mkdir(parents=True)
    source_file = write_file(nested / "charge.txt", "内容")

    document = load_document(source_file, root=tmp_path)

    assert document.source == "raw/manual/charge.txt"
    assert source_name(source_file, root=tmp_path) == "raw/manual/charge.txt"


def test_source_falls_back_to_the_file_name_without_root(tmp_path: Path) -> None:
    source_file = write_file(tmp_path / "manual.txt", "内容")

    assert load_document(source_file).source == "manual.txt"


def test_undecodable_file_is_reported(tmp_path: Path) -> None:
    source_file = tmp_path / "broken.txt"
    source_file.write_bytes(b"\xff\xff\xff")

    with pytest.raises(EncodingDetectionError) as excinfo:
        load_document(source_file, root=tmp_path)

    assert excinfo.value.source == "broken.txt"


def test_empty_and_whitespace_only_documents_are_rejected(tmp_path: Path) -> None:
    empty_file = write_file(tmp_path / "empty.txt", "")
    blank_file = write_file(tmp_path / "blank.txt", "\n   \n\t\n")

    with pytest.raises(EmptyDocumentError):
        load_document(empty_file, root=tmp_path)

    with pytest.raises(EmptyDocumentError):
        load_document(blank_file, root=tmp_path)


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DocumentNotFoundError) as excinfo:
        load_document(tmp_path / "missing.txt", root=tmp_path)

    assert excinfo.value.code == "document_not_found"


def test_unsupported_type_is_reported_with_its_source(tmp_path: Path) -> None:
    source_file = write_file(tmp_path / "data.csv", "a,b")

    with pytest.raises(UnsupportedFileTypeError) as excinfo:
        load_document(source_file, root=tmp_path)

    assert excinfo.value.code == "unsupported_file_type"
    assert excinfo.value.source == "data.csv"


def test_batch_isolates_single_file_failures(tmp_path: Path) -> None:
    good = write_file(tmp_path / "good.txt", "可用资料")
    empty = write_file(tmp_path / "empty.txt", "")
    unsupported = write_file(tmp_path / "data.csv", "a,b")

    batch = load_documents([good, empty, unsupported, tmp_path / "missing.txt"], root=tmp_path)

    assert [document.source for document in batch.documents] == ["good.txt"]
    assert {failure.code for failure in batch.failures} == {
        "empty_document",
        "unsupported_file_type",
        "document_not_found",
    }
    assert batch.total == 4
    assert all(failure.message for failure in batch.failures)


def test_duplicate_content_is_flagged_without_losing_the_copy(tmp_path: Path) -> None:
    first = write_file(tmp_path / "manual.txt", "相同内容")
    second = write_file(tmp_path / "copy.txt", "相同内容")

    batch = load_documents([first, second], root=tmp_path)

    documents = {document.source: document for document in batch.documents}
    assert documents["manual.txt"].duplicate_of is None
    assert documents["manual.txt"].status is DocumentStatus.PARSED
    assert documents["copy.txt"].duplicate_of == "manual.txt"
    assert documents["copy.txt"].status is DocumentStatus.DUPLICATE
    assert batch.duplicate_sources == ("copy.txt",)


def test_iter_supported_files_is_sorted_and_filtered(tmp_path: Path) -> None:
    write_file(tmp_path / "b.md", "b")
    write_file(tmp_path / "a.txt", "a")
    write_file(tmp_path / "skip.csv", "x")
    nested = tmp_path / "nested"
    nested.mkdir()
    write_file(nested / "c.txt", "c")

    found = [path.relative_to(tmp_path).as_posix() for path in iter_supported_files(tmp_path)]

    assert found == ["a.txt", "b.md", "nested/c.txt"]


def test_iter_supported_files_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(DocumentNotFoundError):
        iter_supported_files(tmp_path / "nope")
