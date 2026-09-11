"""The index-like chunk filter used before embedding."""

from __future__ import annotations

import pytest

from rag_agent.ingestion.filters import is_index_like


def entry(title: str, page: int, dots: int = 70) -> str:
    """Build one table-of-contents line with dot leaders."""

    return f"{title} {'.' * dots}{page}"


TABLE_OF_CONTENTS = "\n".join(
    (
        "3",
        "目录",
        entry("1. 产品组成", 4),
        entry(" 1.1 包装内容物", 4),
        entry(" 1.2 部件名称", 4),
        entry("2. 产品使用", 8),
        entry("     2.1 注意事项", 8),
    )
)


def test_table_of_contents_is_detected() -> None:
    assert is_index_like(TABLE_OF_CONTENTS) is True


def test_html_style_dot_leaders_without_numbers_are_kept() -> None:
    prose = (
        "使用产品前请仔细阅读此说明书。\n"
        "若主机在复式楼梯口等位置执行任务，请放置防护栏。\n"
        "请避免站在主机前方，以免主机识别不到待清扫的区域。"
    )

    assert is_index_like(prose) is False


def test_a_single_entry_inside_prose_is_kept() -> None:
    prose = entry("4. 常见问题排查", 23) + "\n" + "正文说明。" * 20

    assert is_index_like(prose) is False


def test_short_chunks_are_never_filtered() -> None:
    assert is_index_like("目录\n" + entry("1. 产品组成", 4, dots=6)) is False


def test_standards_table_is_kept() -> None:
    table = (
        "28\n执行标准：\n"
        "GB4706.1- 2005             GB4706.7  - 2014\n"
        "GB4343.1- 2018             GB17625.1- 2012\n"
        "部件类别\n有害物质\n铅（Pb）\n及其化合物"
    )

    assert is_index_like(table) is False


def test_ratio_is_configurable() -> None:
    mixed = "1. 产品组成 ......4\n2. 产品使用 ......8\n正文说明。\n正文说明。"

    assert is_index_like(mixed, leader_ratio=0.4) is True
    assert is_index_like(mixed, leader_ratio=0.9) is False


def test_min_lines_is_configurable() -> None:
    two_lines = "1. 产品组成 ......4\n2. 产品使用 ......8"

    assert is_index_like(two_lines) is False
    assert is_index_like(two_lines, min_lines=2) is True


def test_blank_content_is_kept() -> None:
    assert is_index_like("") is False
    assert is_index_like("   \n\n  ") is False


@pytest.mark.parametrize(
    "content",
    [
        "1. 产品组成 4\n2. 产品使用 8\n3. 常见问题排查 23",
        "1. 产品组成 - 4\n2. 产品使用 - 8\n3. 常见问题排查 - 23",
    ],
)
def test_entries_without_dot_leaders_are_kept(content: str) -> None:
    assert is_index_like(content) is False
