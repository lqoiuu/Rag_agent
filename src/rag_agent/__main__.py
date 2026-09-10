"""Command-line entry point for the project."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from rag_agent.health import collect_health
from rag_agent.observability.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-agent")
    parser.add_argument(
        "command",
        choices=("health",),
        help="Command to execute.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()

    if args.command == "health":
        report = collect_health()
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return 0 if report.status == "ok" else 1

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
