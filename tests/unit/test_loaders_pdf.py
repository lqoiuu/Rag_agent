"""PDF loading behaviour.

These tests need the optional ``pypdf`` dependency, so the whole module is
skipped when it is not installed.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from pdf_fixtures import build_text_pdf

from rag_agent.domain.documents import FileType
from rag_agent.domain.errors import EmptyDocumentError, EncryptedDocumentError, IngestionError
from rag_agent.ingestion.loaders import load_document

pypdf: Any = pytest.importorskip("pypdf")


def encrypt_pdf(data: bytes, password: str) -> bytes:
    """Return the same PDF protected with a user password."""

    reader = pypdf.PdfReader(io.BytesIO(data))
    writer = pypdf.PdfWriter(clone_from=reader)
    writer.encrypt(password, algorithm="RC4-128")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_pdf_pages_are_extracted_with_page_numbers(tmp_path: Path) -> None:
    source_file = tmp_path / "manual.pdf"
    source_file.write_bytes(build_text_pdf(["Dustbin is stuck", "Charge base is offline"]))

    document = load_document(source_file, root=tmp_path)

    assert document.file_type is FileType.PDF
    assert document.page_count == 2
    assert document.encoding is None
    assert document.pages[0].page_number == 1
    assert "Dustbin is stuck" in document.pages[0].text
    assert "Charge base is offline" in document.pages[1].text
    assert "Dustbin is stuck" in document.text


def test_pdf_without_text_is_rejected(tmp_path: Path) -> None:
    source_file = tmp_path / "blank.pdf"
    source_file.write_bytes(build_text_pdf([""]))

    with pytest.raises(EmptyDocumentError):
        load_document(source_file, root=tmp_path)


def test_encrypted_pdf_is_rejected(tmp_path: Path) -> None:
    source_file = tmp_path / "locked.pdf"
    source_file.write_bytes(encrypt_pdf(build_text_pdf(["secret page"]), "secret"))

    with pytest.raises(EncryptedDocumentError) as excinfo:
        load_document(source_file, root=tmp_path)

    assert excinfo.value.code == "encrypted_document"
    assert excinfo.value.source == "locked.pdf"


def test_corrupted_pdf_is_rejected(tmp_path: Path) -> None:
    source_file = tmp_path / "broken.pdf"
    source_file.write_bytes(b"\x00\x01\x02\x03 definitely not a pdf")

    with pytest.raises(IngestionError) as excinfo:
        load_document(source_file, root=tmp_path)

    # pypdf may report a parse failure or hand back a document without pages.
    assert excinfo.value.code in {"corrupted_document", "empty_document"}
