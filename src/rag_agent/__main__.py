"""Command-line entry point for the project."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from rag_agent.config import build_chat_model, build_embedding_model, get_settings
from rag_agent.config.settings import Settings
from rag_agent.domain.errors import IngestionError
from rag_agent.generation import answer_question
from rag_agent.health import collect_health
from rag_agent.ingestion import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    ChunkingConfig,
    IngestResult,
    ingest_directory,
    ingest_path,
    load_document,
    raw_directory,
    remove_document,
    split_document,
    summarize_chunks,
    sync_index,
)
from rag_agent.observability.logging import configure_logging
from rag_agent.providers.base import EmbeddingModel, ModelError
from rag_agent.retrieval import Retriever, RetrieverConfig
from rag_agent.storage import MetadataStore
from rag_agent.vectorstore import ChunkVectorStore

EXIT_OK = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-agent")
    parser.add_argument(
        "command",
        choices=(
            "health",
            "ask",
            "search",
            "chunk-report",
            "ingest",
            "reindex",
            "delete-document",
        ),
        help="Command to execute.",
    )
    parser.add_argument(
        "argument",
        nargs="?",
        help='Question text for "ask" and "search", a file path for "chunk-report", '
        'a path for "ingest", or a stored source label for "delete-document".',
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index documents even when the stored version is unchanged.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override the configured number of retrieval hits.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the configured cosine similarity threshold.",
    )
    parser.add_argument(
        "--source",
        default="",
        help="Restrict retrieval to one stored source label.",
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
        return EXIT_ERROR

    payload = answer.model_dump()
    payload["latency_ms"] = round(answer.latency_ms, 1)
    print(json.dumps({"status": "ok", **payload}, ensure_ascii=False, indent=2))
    return EXIT_OK


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
        return EXIT_ERROR
    except (OSError, ValueError) as exc:
        _error_payload("invalid_argument", str(exc))
        return EXIT_ERROR

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
    return EXIT_OK


@contextmanager
def _open_index(
    settings: Settings,
) -> Iterator[tuple[MetadataStore, ChunkVectorStore, EmbeddingModel]]:
    """Open the metadata store and the vector store for one command."""

    with MetadataStore(settings.sqlite_path) as store:
        vectors = ChunkVectorStore(settings.chroma_dir)
        with build_embedding_model(settings) as model:
            yield store, vectors, model


@contextmanager
def _open_retrieval(
    settings: Settings,
) -> Iterator[tuple[ChunkVectorStore, EmbeddingModel]]:
    """Open only what retrieval needs: vectors plus an embedding model."""

    vectors = ChunkVectorStore(settings.chroma_dir)
    with build_embedding_model(settings) as model:
        yield vectors, model


def _search(query: str, *, top_k: int | None, threshold: float | None, source: str) -> int:
    """Run one retrieval query and report why it hit or refused to trust it."""

    settings = get_settings()
    try:
        with _open_retrieval(settings) as (vectors, model):
            retriever = Retriever(
                vectors=vectors,
                embedding_model=model,
                config=RetrieverConfig(
                    top_k=settings.retrieval_top_k,
                    threshold=settings.retrieval_threshold,
                ),
            )
            result = retriever.search(
                query,
                top_k=top_k,
                threshold=threshold,
                source=source or None,
            )
    except ModelError as exc:
        _error_payload(exc.code, str(exc))
        return EXIT_ERROR
    except ValueError as exc:
        _error_payload("invalid_argument", str(exc))
        return EXIT_ERROR

    print(json.dumps({"status": "ok", **result.as_dict()}, ensure_ascii=False, indent=2))
    return EXIT_OK if result.is_confident else EXIT_PARTIAL_FAILURE


def _resolved_root(path: Path, settings: Settings) -> Path | None:
    """Use the raw directory as source root only when the file lives inside it."""

    raw_dir = raw_directory(settings.data_dir)
    try:
        path.resolve().relative_to(raw_dir.resolve())
    except ValueError:
        return None
    return raw_dir


def _report_results(payload: dict[str, object], results: Sequence[IngestResult]) -> int:
    failures = [result for result in results if not result.ok]
    print(
        json.dumps(
            {
                "status": "ok" if not failures else "partial",
                **payload,
                "results": [result.as_dict() for result in results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return EXIT_OK if not failures else EXIT_PARTIAL_FAILURE


def _ingest(path: str, *, force: bool) -> int:
    target = Path(path)
    settings = get_settings()
    try:
        with _open_index(settings) as (store, vectors, model):
            if target.is_dir():
                results = ingest_directory(
                    target, store=store, vectors=vectors, embedding_model=model, force=force
                )
            else:
                results = (
                    ingest_path(
                        target,
                        store=store,
                        vectors=vectors,
                        embedding_model=model,
                        root=_resolved_root(target, settings),
                        force=force,
                    ),
                )
    except ModelError as exc:
        _error_payload(exc.code, str(exc))
        return EXIT_ERROR
    except IngestionError as exc:
        _error_payload(exc.code, str(exc))
        return EXIT_ERROR

    return _report_results({"target": str(target)}, results)


def _reindex() -> int:
    settings = get_settings()
    raw_dir = raw_directory(settings.data_dir)
    try:
        with _open_index(settings) as (store, vectors, model):
            results = sync_index(raw_dir, store=store, vectors=vectors, embedding_model=model)
    except ModelError as exc:
        _error_payload(exc.code, str(exc))
        return EXIT_ERROR
    except IngestionError as exc:
        _error_payload(exc.code, str(exc))
        return EXIT_ERROR

    return _report_results({"target": str(raw_dir)}, results)


def _delete_document(source: str) -> int:
    settings = get_settings()
    try:
        with _open_index(settings) as (store, vectors, _model):
            result = remove_document(source, store=store, vectors=vectors)
    except ModelError as exc:
        _error_payload(exc.code, str(exc))
        return EXIT_ERROR
    except IngestionError as exc:
        _error_payload(exc.code, str(exc))
        return EXIT_ERROR

    return _report_results({"target": source}, (result,))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging()

    if args.command == "health":
        report = collect_health()
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return EXIT_OK if report.status == "ok" else EXIT_PARTIAL_FAILURE

    if args.command == "ask":
        question = (args.argument or "").strip()
        if not question:
            parser.error('ask requires a question, for example: rag-agent ask "问题"')
        return _ask(question)

    if args.command == "search":
        query = (args.argument or "").strip()
        if not query:
            parser.error('search requires a query, for example: rag-agent search "主刷卡住"')
        return _search(
            query,
            top_k=args.top_k,
            threshold=args.threshold,
            source=args.source,
        )

    if args.command == "chunk-report":
        path = (args.argument or "").strip()
        if not path:
            parser.error("chunk-report requires a file path")
        return _chunk_report(path, args.chunk_sizes, args.chunk_overlaps)

    if args.command == "ingest":
        path = (args.argument or "").strip()
        if not path:
            parser.error("ingest requires a file or directory path")
        return _ingest(path, force=args.force)

    if args.command == "reindex":
        return _reindex()

    if args.command == "delete-document":
        source = (args.argument or "").strip()
        if not source:
            parser.error("delete-document requires the stored source label")
        return _delete_document(source)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
