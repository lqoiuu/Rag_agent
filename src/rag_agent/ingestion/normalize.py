"""Decoding and text normalization shared by every document loader."""

from __future__ import annotations

import unicodedata
from codecs import BOM_UTF8

from rag_agent.domain.errors import EncodingDetectionError

CANDIDATE_ENCODINGS: tuple[str, ...] = ("utf-8", "gb18030", "big5")

_KEPT_CONTROL_CHARACTERS = frozenset({"\t"})


def decode_bytes(data: bytes, *, source: str | None = None) -> tuple[str, str]:
    """Decode raw bytes with the first candidate encoding that succeeds.

    A UTF-8 byte order mark is detected explicitly so plain UTF-8 files are not
    mislabeled, while ``gb18030`` (a superset of GBK and GB2312) and ``big5``
    cover legacy Chinese documents. Detection is heuristic: candidates are tried
    in a fixed order, so bytes that are valid in more than one legacy encoding
    resolve to the earlier candidate.
    """

    if not data:
        return "", "utf-8"

    if data.startswith(BOM_UTF8):
        return data.decode("utf-8-sig"), "utf-8-sig"

    for encoding in CANDIDATE_ENCODINGS:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    tried = ", ".join(("utf-8-sig", *CANDIDATE_ENCODINGS))
    raise EncodingDetectionError(f"could not decode document; tried: {tried}", source=source)


def normalize_text(text: str) -> str:
    """Normalize line endings, control characters and blank-line runs.

    The function is idempotent: normalizing an already normalized text returns
    the same string, which keeps hashing and chunking reproducible.
    """

    if not text:
        return ""

    unified = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    cleaned = [_clean_line(line) for line in unified.split("\n")]
    return "\n".join(_collapse_blank_lines(cleaned)).strip()


def is_blank(text: str) -> bool:
    """True when the text contains no visible content."""

    return not text.strip()


def _clean_line(line: str) -> str:
    kept = "".join(
        character
        for character in line
        if character in _KEPT_CONTROL_CHARACTERS or not _is_control(character)
    )
    return kept.rstrip()


def _is_control(character: str) -> bool:
    return unicodedata.category(character) in {"Cc", "Cf"}


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    collapsed: list[str] = []
    for line in lines:
        if line or (collapsed and collapsed[-1]):
            collapsed.append(line)
    return collapsed
