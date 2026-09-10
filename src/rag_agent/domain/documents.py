"""Domain objects describing source documents and their retrievable chunks.

Two identities are deliberately kept apart:

- ``document_id`` is derived from the source path only, so it survives content
  edits and later stages can find the chunks that a new version replaces.
- ``checksum`` and ``version`` mirror the content, so a changed file is a new
  version of the same logical document.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from rag_agent.domain.errors import UnsupportedFileTypeError


class FileType(StrEnum):
    """Source file types the project can load."""

    TXT = "txt"
    MARKDOWN = "markdown"
    PDF = "pdf"


class DocumentStatus(StrEnum):
    """Outcome of loading one source document."""

    PARSED = "parsed"
    DUPLICATE = "duplicate"


SUFFIX_TO_FILE_TYPE: dict[str, FileType] = {
    ".txt": FileType.TXT,
    ".text": FileType.TXT,
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".pdf": FileType.PDF,
}


def file_type_for_suffix(suffix: str, *, source: str | None = None) -> FileType:
    """Map a file suffix to a supported type, or fail with a stable code."""

    file_type = SUFFIX_TO_FILE_TYPE.get(suffix.lower())
    if file_type is None:
        supported = ", ".join(sorted(SUFFIX_TO_FILE_TYPE))
        raise UnsupportedFileTypeError(
            f"unsupported file type {suffix!r}; supported: {supported}", source=source
        )
    return file_type


def build_document_id(source: str) -> str:
    """Derive the stable logical document ID from the source label only."""

    return f"doc-{hashlib.sha256(source.encode('utf-8')).hexdigest()[:16]}"


def checksum_of(data: bytes) -> str:
    """Return the SHA-256 checksum of raw file bytes."""

    return hashlib.sha256(data).hexdigest()


def version_of(checksum: str) -> str:
    """Derive the short content version used for reindex decisions."""

    return checksum[:12]


def build_chunk_id(document_id: str, index: int) -> str:
    """Derive a deterministic chunk ID from document identity and order."""

    return f"{document_id}-c{index:04d}"


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """Text extracted from one page or one logical section."""

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A loaded, normalized source document."""

    document_id: str
    source: str
    file_type: FileType
    checksum: str
    version: str
    pages: tuple[DocumentPage, ...]
    size_bytes: int
    encoding: str | None = None
    duplicate_of: str | None = None

    @property
    def status(self) -> DocumentStatus:
        """Derived status: a document is a duplicate when a copy came first."""

        return DocumentStatus.DUPLICATE if self.duplicate_of else DocumentStatus.PARSED

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        """Whole document text with a blank line between pages."""

        return "\n\n".join(page.text for page in self.pages)


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A retrievable slice of a source document.

    Chunking itself belongs to the next stage; this model freezes the fields
    that retrieval, citations and evaluation will depend on.
    """

    chunk_id: str
    document_id: str
    source: str
    content: str
    index: int
    char_range: tuple[int, int]
    page: int | None = None
    heading: str | None = None
