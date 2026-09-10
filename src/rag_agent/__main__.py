"""Command-line entry point for the project."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from rag_agent.config import build_chat_model
from rag_agent.generation import answer_question
from rag_agent.health import collect_health
from rag_agent.observability.logging import configure_logging
from rag_agent.providers.base import ModelError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-agent")
    parser.add_argument(
        "command",
        choices=("health", "ask"),
        help="Command to execute.",
    )
    parser.add_argument(
        "argument",
        nargs="?",
        help='Question text for the "ask" command.',
    )
    return parser


def _ask(question: str) -> int:
    """Answer one question through the configured provider layer."""

    try:
        with build_chat_model() as model:
            answer = answer_question(model, question)
    except ModelError as exc:
        print(
            json.dumps(
                {"status": "error", "code": exc.code, "message": str(exc)},
                ensure_ascii=False,
            )
        )
        return 2

    payload = answer.model_dump()
    payload["latency_ms"] = round(answer.latency_ms, 1)
    print(json.dumps({"status": "ok", **payload}, ensure_ascii=False, indent=2))
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

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
