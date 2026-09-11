"""Content filters applied before chunks are embedded.

The first rule targets table-of-contents and index pages. Stage 6 measured such
a chunk outranking the chunk that actually contains the answer (0.6318 versus
0.6307), and a listing page by itself carries no answerable fact: it is a list
of titles plus dot leaders and page numbers. The rule is deterministic and can
be switched off, so stage 8 can measure the effect instead of trusting it.
"""

from __future__ import annotations

import re

DEFAULT_LEADER_RATIO = 0.3
DEFAULT_MIN_LINES = 3

_LEADER_LINE = re.compile(r"\.{3,}\s*\d+\s*$")


def is_index_like(
    content: str,
    *,
    leader_ratio: float = DEFAULT_LEADER_RATIO,
    min_lines: int = DEFAULT_MIN_LINES,
) -> bool:
    """True when a chunk looks like a table of contents or an index listing.

    A line counts as an entry when it ends with a run of dots followed by a page
    number, for example ``1. 产品组成 ........ 4``. Short chunks never match, so a
    paragraph that happens to mention one entry is kept.
    """

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) < min_lines:
        return False
    entries = sum(1 for line in lines if _LEADER_LINE.search(line))
    return entries / len(lines) >= leader_ratio
