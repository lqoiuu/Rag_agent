"""Turn supported files into :class:`SourceDocument` objects.

Loading is deliberately failure-isolated: one broken file must never stop a
batch, and every failure keeps a stable code plus the source it came from.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rag_agent.domain.documents import (
    SUFFIX_TO_FILE_TYPE,
    DocumentPage,
    FileType,
    SourceDocument,
    build_document_id,
    checksum_of,
    file_type_for_suffix,
    version_of,
)
from rag_agent.domain.errors import (
    CorruptedDocumentError,
    DocumentNotFoundError,
    EmptyDocumentError,
    EncryptedDocumentError,
    IngestionError,
    MissingDependencyError,
)
from rag_agent.ingestion.normalize import decode_bytes, is_blank, normalize_text

PDF_DEPENDENCY = "pypdf"


@dataclass(frozen=True, slots=True)
class LoadFailure:
    """One file that could not be loaded."""

    source: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class LoadBatch:
    """Outcome of loading many files at once."""

    documents: tuple[SourceDocument, ...]
    failures: tuple[LoadFailure, ...]

    @property
    def total(self) -> int:
        return len(self.documents) + len(self.failures)

    @property
    def duplicate_sources(self) -> tuple[str, ...]:
        return tuple(document.source for document in self.documents if document.duplicate_of)


def source_name(path: Path, *, root: Path | None = None) -> str:
    """Return the stable source label stored on a document.

    Paths are recorded relative to ``root`` when possible so identities stay
    reproducible across machines; otherwise the bare file name is used.
    """

    resolved = path.resolve()
    if root is not None:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            return resolved.name
    return resolved.name


def iter_supported_files(root: Path | str) -> tuple[Path, ...]:
    """Return supported files under ``root`` in deterministic order."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise DocumentNotFoundError(f"directory does not exist: {root_path}", source=str(root_path))
    return tuple(
        path
        for path in sorted(root_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUFFIX_TO_FILE_TYPE
    )


def load_document(path: Path | str, *, root: Path | None = None) -> SourceDocument:
    """Load one file into a normalized, traceable source document."""

    file_path = Path(path)
    source = source_name(file_path, root=root)

    if not file_path.is_file():
        raise DocumentNotFoundError(f"file does not exist: {source}", source=source)

    file_type = file_type_for_suffix(file_path.suffix, source=source)
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise IngestionError(f"cannot read file: {exc}", source=source) from exc

    if file_type is FileType.PDF:
        pages = _pdf_pages(file_path, source=source)
        encoding: str | None = None
    else:
        text, encoding = decode_bytes(data, source=source)
        pages = (DocumentPage(page_number=1, text=normalize_text(text)),)

    if all(is_blank(page.text) for page in pages):
        raise EmptyDocumentError("document contains no extractable text", source=source)

    checksum = checksum_of(data)
    return SourceDocument(
        document_id=build_document_id(source),
        source=source,
        file_type=file_type,
        checksum=checksum,
        version=version_of(checksum),
        pages=pages,
        size_bytes=len(data),
        encoding=encoding,
    )


def load_documents(paths: Iterable[Path | str], *, root: Path | None = None) -> LoadBatch:
    """Load many files, isolating failures and flagging duplicate content."""

    documents: list[SourceDocument] = []
    failures: list[LoadFailure] = []
    seen_checksums: dict[str, str] = {}

    for path in paths:
        try:
            document = load_document(path, root=root)
        except IngestionError as exc:
            failures.append(
                LoadFailure(source=exc.source or str(path), code=exc.code, message=str(exc))
            )
            continue

        duplicate_of = seen_checksums.get(document.checksum)
        if duplicate_of is None:
            seen_checksums[document.checksum] = document.source
        else:
            document = replace(document, duplicate_of=duplicate_of)
        documents.append(document)

    return LoadBatch(documents=tuple(documents), failures=tuple(failures))


def _pdf_pages(path: Path, *, source: str) -> tuple[DocumentPage, ...]:
    pypdf: Any = _import_pdf_library(source=source)
    reader = _open_pdf_reader(pypdf, path, source=source)
    pages: list[DocumentPage] = []
    for number, page in enumerate(_reader_pages(reader, source=source), start=1):
        pages.append(
            DocumentPage(
                page_number=number,
                text=normalize_text(_extract_text(page, source=source)),
            )
        )
    return tuple(pages)


def _import_pdf_library(*, source: str) -> Any:
    """Import pypdf lazily so text loaders work without it installed.

    The module is intentionally treated as ``Any``: pypdf stays an optional
    parser dependency, and static checks of this module must not depend on its
    stubs being present in the analysing environment.
    """

    try:
        return importlib.import_module(PDF_DEPENDENCY)
    except ImportError as exc:
        raise MissingDependencyError(
            f"PDF support requires the {PDF_DEPENDENCY} package; run uv sync to install it",
            source=source,
        ) from exc


def _open_pdf_reader(pypdf: Any, path: Path, *, source: str) -> Any:
    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as exc:  # pypdf raises several unrelated error types
        raise CorruptedDocumentError(f"cannot open PDF: {exc}", source=source) from exc

    if getattr(reader, "is_encrypted", False) and not reader.decrypt(""):
        raise EncryptedDocumentError(
            "PDF is encrypted and needs a password the project does not store", source=source
        )
    return reader


def _reader_pages(reader: Any, *, source: str) -> list[Any]:
    try:
        return list(reader.pages)
    except Exception as exc:  # pypdf raises several unrelated error types
        raise CorruptedDocumentError(f"cannot read PDF pages: {exc}", source=source) from exc


def _extract_text(page: Any, *, source: str) -> str:
    try:
        text: str = page.extract_text() or ""
    except Exception as exc:  # pypdf raises several unrelated error types
        raise CorruptedDocumentError(f"cannot extract PDF text: {exc}", source=source) from exc
    return text
