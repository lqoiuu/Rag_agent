"""Structured logging behaviour."""

from __future__ import annotations

import json
import logging

from rag_agent.observability.logging import JsonFormatter, configure_logging


def test_configure_logging_installs_one_json_handler() -> None:
    configure_logging("DEBUG")

    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG
    assert len(root_logger.handlers) == 1
    assert isinstance(root_logger.handlers[0].formatter, JsonFormatter)

    configure_logging()  # 重复调用不应堆积 handler
    assert len(logging.getLogger().handlers) == 1


def test_configure_logging_quiets_the_http_client() -> None:
    configure_logging("INFO")

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("rag_agent").getEffectiveLevel() == logging.INFO


def test_json_formatter_emits_one_object_per_record() -> None:
    record = logging.LogRecord(
        name="rag_agent.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="model=%s attempts=%d",
        args=("qwen-plus", 2),
        exc_info=None,
    )

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "rag_agent.test"
    assert payload["message"] == "model=qwen-plus attempts=2"
    assert "timestamp" in payload
    assert "exception" not in payload


def test_json_formatter_includes_exception_details() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="rag_agent.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="failed",
        args=(),
        exc_info=exc_info,
    )

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: boom" in payload["exception"]
