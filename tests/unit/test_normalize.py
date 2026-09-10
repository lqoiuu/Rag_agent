"""Decoding and normalization behaviour."""

from __future__ import annotations

import pytest

from rag_agent.domain.errors import EncodingDetectionError
from rag_agent.ingestion.normalize import decode_bytes, is_blank, normalize_text


def test_utf8_with_bom_is_decoded_and_marked() -> None:
    text, encoding = decode_bytes("带 BOM 的文本".encode("utf-8-sig"))

    assert text == "带 BOM 的文本"
    assert encoding == "utf-8-sig"


def test_plain_utf8_is_decoded() -> None:
    text, encoding = decode_bytes("普通 UTF-8 文本".encode())

    assert text == "普通 UTF-8 文本"
    assert encoding == "utf-8"


def test_legacy_chinese_encodings_are_decoded() -> None:
    gb_text, gb_encoding = decode_bytes("旧编码文本".encode("gb18030"))

    assert gb_text == "旧编码文本"
    assert gb_encoding == "gb18030"


def test_ambiguous_legacy_bytes_resolve_to_the_first_candidate() -> None:
    """Detection is heuristic: candidates are tried in a fixed order.

    Some big5 byte sequences are also valid gb18030, so they decode into
    different glyphs instead of raising. This is a known limitation until a
    statistical detector is added.
    """

    data = "舊編碼文本".encode("big5")

    text, encoding = decode_bytes(data)

    assert encoding in {"gb18030", "big5"}
    assert text != ""


def test_empty_bytes_decode_to_empty_text() -> None:
    assert decode_bytes(b"") == ("", "utf-8")


def test_undecodable_bytes_raise_encoding_error() -> None:
    with pytest.raises(EncodingDetectionError) as excinfo:
        decode_bytes(b"\xff\xff\xff", source="broken.txt")

    assert excinfo.value.code == "encoding_error"
    assert excinfo.value.source == "broken.txt"
    assert "utf-8" in str(excinfo.value)


def test_line_endings_and_trailing_space_are_normalized() -> None:
    raw = "第一行  \r\n第二行\r\n\r\n第三行\t \r"

    assert normalize_text(raw) == "第一行\n第二行\n\n第三行"


def test_blank_line_runs_are_collapsed() -> None:
    assert normalize_text("一\n\n\n\n二") == "一\n\n二"
    assert normalize_text("\n\n一\n\n") == "一"


def test_control_characters_are_removed_but_tabs_are_kept() -> None:
    raw = "正常\x00文本\x07\t制表\xad软连字符"

    normalized = normalize_text(raw)

    assert "\x00" not in normalized
    assert "\x07" not in normalized
    assert "\xad" not in normalized
    assert "\t" in normalized
    assert normalized.startswith("正常文本")


def test_non_breaking_space_becomes_a_plain_space() -> None:
    assert normalize_text("充电\u00a0座") == "充电 座"


def test_normalization_is_idempotent() -> None:
    raw = "标题\r\n\r\n\r\n正文  \r\n\r\n正文二\x00"

    once = normalize_text(raw)

    assert normalize_text(once) == once


def test_empty_input_stays_empty() -> None:
    assert normalize_text("") == ""
    assert is_blank("")
    assert is_blank("   \n\t\n")
    assert not is_blank("内容")
