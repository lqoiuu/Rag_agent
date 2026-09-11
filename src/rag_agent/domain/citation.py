"""Citations that tie an answer back to the chunks it actually used."""

from __future__ import annotations

from dataclasses import dataclass

from rag_agent.domain.documents import DocumentChunk

DEFAULT_EXCERPT_CHARS = 160


@dataclass(frozen=True, slots=True)
class Citation:
    """One numbered source used by an answer."""

    citation_id: int
    chunk_id: str
    document_id: str
    source: str
    page: int | None
    heading: str | None
    char_range: tuple[int, int]
    excerpt: str

    @classmethod
    def from_chunk(
        cls,
        citation_id: int,
        chunk: DocumentChunk,
        *,
        excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
    ) -> Citation:
        return cls(
            citation_id=citation_id,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source=chunk.source,
            page=chunk.page,
            heading=chunk.heading,
            char_range=chunk.char_range,
            excerpt=build_excerpt(chunk.content, limit=excerpt_chars),
        )

    @property
    def label(self) -> str:
        """Human-readable location, for example ``manual.pdf · 第 32 页``."""

        parts = [self.source]
        if self.page is not None:
            parts.append(f"第 {self.page} 页")
        if self.heading:
            parts.append(self.heading)
        return " · ".join(parts)

    def as_dict(self) -> dict[str, object]:
        return {
            "citation_id": self.citation_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "page": self.page,
            "heading": self.heading,
            "char_range": list(self.char_range),
            "excerpt": self.excerpt,
            "label": self.label,
        }


def build_excerpt(content: str, *, limit: int = DEFAULT_EXCERPT_CHARS) -> str:
    """Collapse whitespace and shorten content for display."""

    collapsed = " ".join(content.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"
