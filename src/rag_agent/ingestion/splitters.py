"""Heading-aware, recursive character chunking.

Splitting is expressed as an injected :data:`TextSplitter` callable. The tracing
logic — heading hierarchy, page binding, character ranges and empty-piece
filtering — therefore stays testable without any splitter package, and a future
splitter can be swapped in without touching this module.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from rag_agent.domain.documents import (
    DocumentChunk,
    DocumentPage,
    SourceDocument,
    build_chunk_id,
)
from rag_agent.domain.errors import ChunkingError, MissingDependencyError

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "。", "；", " ", "")
SPLITTER_DEPENDENCY = "langchain_text_splitters"

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_HEADING_JOINER = " > "
_PAGE_JOINER = "\n\n"

TextSplitter = Callable[[str], Sequence[str]]


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Parameters of one chunking experiment."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    separators: tuple[str, ...] = DEFAULT_SEPARATORS

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not self.separators:
            raise ValueError("separators must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": list(self.separators),
        }


@dataclass(frozen=True, slots=True)
class TextSection:
    """A heading-delimited region inside one page."""

    heading: str | None
    heading_level: int
    start: int
    text: str


@dataclass(frozen=True, slots=True)
class _Piece:
    content: str
    start: int
    end: int
    heading: str | None


def build_langchain_splitter(config: ChunkingConfig, *, source: str | None = None) -> TextSplitter:
    """Return the default recursive character splitter backed by LangChain.

    ``langchain_text_splitters`` is imported lazily so the rest of this module
    keeps working when the splitter package is not installed.
    """

    try:
        module: Any = importlib.import_module(SPLITTER_DEPENDENCY)
    except ImportError as exc:
        raise MissingDependencyError(
            f"chunking requires the {SPLITTER_DEPENDENCY} package; run uv sync to install it",
            source=source,
        ) from exc

    splitter: Any = module.RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=list(config.separators),
        keep_separator=True,
        strip_whitespace=True,
        length_function=len,
    )
    split_text: TextSplitter = splitter.split_text
    return split_text


def split_into_sections(page_text: str) -> tuple[TextSection, ...]:
    """Split one page into heading-delimited sections.

    Markdown ATX headings (``#`` through ``######``) open a section and are kept
    inside the following section text, so a heading is never separated from the
    content it introduces. Text before the first heading becomes an untitled
    section, and the heading path keeps its hierarchy.
    """

    sections: list[TextSection] = []
    stack: list[tuple[int, str]] = []
    current_start = 0
    current_heading: str | None = None
    current_level = 0
    offset = 0

    for line in page_text.split("\n"):
        match = _HEADING_PATTERN.match(line)
        if match is not None:
            body = page_text[current_start:offset]
            if body.strip():
                sections.append(
                    TextSection(
                        heading=current_heading,
                        heading_level=current_level,
                        start=current_start,
                        text=body,
                    )
                )
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_heading = _HEADING_JOINER.join(item[1] for item in stack)
            current_level = level
            current_start = offset
        offset += len(line) + 1

    tail = page_text[current_start:]
    if tail.strip():
        sections.append(
            TextSection(
                heading=current_heading,
                heading_level=current_level,
                start=current_start,
                text=tail,
            )
        )
    return tuple(sections)


def split_document(
    document: SourceDocument,
    config: ChunkingConfig | None = None,
    *,
    split_text: TextSplitter | None = None,
) -> tuple[DocumentChunk, ...]:
    """Split a loaded document into chunks that stay traceable to their origin.

    ``char_range`` is expressed against ``document.text``, so every chunk can be
    located in the whole document, not only inside its own page.
    """

    resolved = config if config is not None else ChunkingConfig()
    splitter = (
        split_text
        if split_text is not None
        else build_langchain_splitter(resolved, source=document.source)
    )

    chunks: list[DocumentChunk] = []
    page_start = 0
    index = 1

    for page in document.pages:
        for piece in _split_page(page, resolved, splitter, source=document.source):
            chunks.append(
                DocumentChunk(
                    chunk_id=build_chunk_id(document.document_id, index),
                    document_id=document.document_id,
                    source=document.source,
                    content=piece.content,
                    index=index,
                    char_range=(page_start + piece.start, page_start + piece.end),
                    page=page.page_number,
                    heading=piece.heading,
                )
            )
            index += 1
        page_start += len(page.text) + len(_PAGE_JOINER)

    return tuple(chunks)


def _split_page(
    page: DocumentPage,
    config: ChunkingConfig,
    splitter: TextSplitter,
    *,
    source: str,
) -> tuple[_Piece, ...]:
    pieces: list[_Piece] = []

    for section in split_into_sections(page.text):
        previous_end = section.start
        for raw in splitter(section.text):
            content = raw.strip()
            if not content:
                continue
            # Overlapping pieces start before the previous end, so the search
            # window has to step back by at most chunk_overlap.
            search_from = max(section.start, previous_end - config.chunk_overlap)
            start = page.text.find(content, search_from)
            if start < 0:
                raise ChunkingError(
                    "splitter returned text that is not part of the source page",
                    source=source,
                )
            end = start + len(content)
            previous_end = end
            pieces.append(_Piece(content=content, start=start, end=end, heading=section.heading))

    return tuple(pieces)
