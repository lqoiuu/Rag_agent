"""Build tiny but valid PDF files for loader tests.

The builder assembles objects and a correct cross-reference table itself, so the
tests do not depend on pypdf being able to repair a broken file.
"""

from __future__ import annotations

from collections.abc import Sequence


def build_text_pdf(pages: Sequence[str]) -> bytes:
    """Return a minimal single-font PDF with one text page per input string."""

    page_count = len(pages)
    font_object = 3 + 2 * page_count
    objects: list[bytes] = []

    kids = " ".join(f"{2 + index} 0 R" for index in range(1, page_count + 1))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("ascii"))

    for index in range(1, page_count + 1):
        page = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_object} 0 R >> >> "
            f"/Contents {2 + page_count + index} 0 R >>"
        )
        objects.append(page.encode("ascii"))

    for index, text in enumerate(pages, start=1):
        stream = f"BT /F1 12 Tf 72 {700 - 12 * index} Td ({_escape(text)}) Tj ET".encode("ascii")
        header = f"<< /Length {len(stream)} >>".encode("ascii")
        objects.append(header + b"\nstream\n" + stream + b"\nendstream")

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"

    xref_offset = len(out)
    size = len(objects) + 1
    out += f"xref\n0 {size}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    trailer = f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    out += trailer.encode("ascii")
    return bytes(out)


def _escape(text: str) -> str:
    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    return escaped.encode("ascii", "replace").decode("ascii")
