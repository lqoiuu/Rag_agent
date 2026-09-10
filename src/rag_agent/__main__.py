"""Command-line entry point for the project."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from rag_agent.config import build_chat_model
from rag_agent.domain.errors import IngestionError
from rag_agent.generation import answer_question
from rag_agent.health import collect_health
from rag_agent.ingestion import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    ChunkingConfig,
    load_document,
    split_document,
    summarize_chunks,
)
from rag_agent.observability.logging import configure_logging
from rag_agent.providers.base import ModelError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-agent")
    parser.add_argument(
        "command",
        choices=("health", "ask", "chunk-report"),
        help="Command to execute.",
    )
    parser.add_argument(
        "argument",
        nargs="?",
        help='Question text for "ask", or a file path for "chunk-report".',
    )
    parser.add_argument(
        "--chunk-sizes",
        default="",
        help="Comma separated chunk sizes for chunk-report, for example 400,800.",
    )
    parser.add_argument(
        "--chunk-overlaps",
        default="",
        help="Comma separated chunk overlaps for chunk-report, for example 0,80.",
    )
    return parser


def _error_payload(code: str, message: str) -> None:
    print(json.dumps({"status": "error", "code": code, "message": message}, ensure_ascii=False))


def _ask(question: str) -> int:
    """Answer one question through the configured provider layer."""

    try:
        with build_chat_model() as model:
            answer = answer_question(model, question)
    except ModelError as exc:
        _error_payload(exc.code, str(exc))
        return 2

    payload = answer.model_dump()
    payload["latency_ms"] = round(answer.latency_ms, 1)
    print(json.dumps({"status": "ok", **payload}, ensure_ascii=False, indent=2))
    return 0


def _parse_values(raw: str, default: int) -> tuple[int, ...]:
    if not raw.strip():
        return (default,)
    return tuple(int(part) for part in raw.split(",") if part.strip())


def _chunk_report(path: str, chunk_sizes: str, chunk_overlaps: str) -> int:
    """Report chunk count and length distribution for one or more settings."""

    try:
        sizes = _parse_values(chunk_sizes, DEFAULT_CHUNK_SIZE)
        overlaps = _parse_values(chunk_overlaps, DEFAULT_CHUNK_OVERLAP)
        configs = [
            ChunkingConfig(chunk_size=size, chunk_overlap=overlap)
            for size in sizes
            for overlap in overlaps
        ]
        document = load_document(path)
        experiments = [
            {**config.as_dict(), **summarize_chunks(split_document(document, config)).as_dict()}
            for config in configs
        ]
    except IngestionError as exc:
        _error_payload(exc.code, str(exc))
        return 2
    except (OSError, ValueError) as exc:
        _error_payload("invalid_argument", str(exc))
        return 2

    print(
        json.dumps(
            {
                "status": "ok",
                "source": document.source,
                "document_id": document.document_id,
                "page_count": document.page_count,
                "experiments": experiments,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging()

    if args.command == "health":
        report = collect_health()
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return 0 if report.status == "ok" else 1

    if args.command == "ask":
        question = (args.argument or "").strip()
        if not question:
            parser.error('ask requires a question, for example: rag-agent ask "问题"')
        return _ask(question)

    if args.command == "chunk-report":
        path = (args.argument or "").strip()
        if not path:
            parser.error("chunk-report requires a file path")
        return _chunk_report(path, args.chunk_sizes, args.chunk_overlaps)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
