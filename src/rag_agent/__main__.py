"""Command-line entry point for the project."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rag_agent.agent import (
    STATUS_ANSWERED,
    STATUS_TICKET_CANCELLED,
    STATUS_TICKET_CREATED,
    ChatTurn,
    build_agent_graph,
    chat_turn,
    run_agent,
)
from rag_agent.config import build_chat_model, build_embedding_model, get_settings
from rag_agent.config.settings import Settings
from rag_agent.domain.errors import IngestionError
from rag_agent.evaluation import (
    EvaluationDatasetError,
    load_cases,
    run_answer_evaluation,
    run_retrieval_evaluation,
    to_json,
    to_markdown,
)
from rag_agent.generation import (
    RagAnswer,
    answer_question,
    answer_with_context,
    finalize_answer,
    prepare_answer,
    stream_raw_answer,
)
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
from rag_agent.memory import (
    ConversationStore,
    ConversationWindow,
    SQLiteCheckpointer,
    ThreadOwnershipError,
    render_prompt_context,
)
from rag_agent.observability.logging import configure_logging
from rag_agent.providers.base import ChatModel, EmbeddingModel, ModelError
from rag_agent.retrieval import Retriever, RetrieverConfig
from rag_agent.storage import BusinessRepository, MetadataStore
from rag_agent.tools import (
    DEVICE_LOOKUP,
    ORDER_LOOKUP,
    TICKET_CREATE,
    TOOL_NAMES,
    USER_LOOKUP,
    CreateTicketArgs,
    DeviceLookupArgs,
    OrderLookupArgs,
    UserLookupArgs,
    create_ticket,
    device_lookup,
    order_lookup,
    user_lookup,
)
from rag_agent.vectorstore import ChunkVectorStore

EXIT_OK = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_ERROR = 2

TOOL_ARG_MODELS: dict[str, Any] = {
    USER_LOOKUP: UserLookupArgs,
    DEVICE_LOOKUP: DeviceLookupArgs,
    ORDER_LOOKUP: OrderLookupArgs,
    TICKET_CREATE: CreateTicketArgs,
}

TOOL_FUNCTIONS: dict[str, Any] = {
    USER_LOOKUP: user_lookup,
    DEVICE_LOOKUP: device_lookup,
    ORDER_LOOKUP: order_lookup,
    TICKET_CREATE: create_ticket,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-agent")
    parser.add_argument(
        "command",
        choices=(
            "health",
            "ask",
            "answer",
            "search",
            "chunk-report",
            "ingest",
            "reindex",
            "delete-document",
            "evaluate",
            "tool",
            "agent",
            "chat",
            "thread",
        ),
        help="Command to execute.",
    )
    parser.add_argument(
        "argument",
        nargs="?",
        help='Question text for "ask", "answer", "search", "agent" and "chat"; a file path for '
        '"chunk-report"; a path for "ingest"; a stored source label for "delete-document"; '
        'or "list" / "clear" / "preferences" for "thread".',
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
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the raw model output for the answer command before the JSON result.",
    )
    parser.add_argument(
        "--keep-index-chunks",
        action="store_true",
        help="Keep table-of-contents style chunks when indexing instead of dropping them.",
    )
    parser.add_argument(
        "--mode",
        choices=("retrieval", "answer"),
        default="retrieval",
        help="Evaluation mode: retrieval only, or retrieval plus grounded answering.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Path to the evaluation set; defaults to data/eval/qa_set.jsonl.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Directory for the generated reports; defaults to data/eval/reports.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases, for a quick check.",
    )
    parser.add_argument(
        "--args",
        default="",
        help='JSON arguments for "tool", for example --args "{\\"user_id\\": \\"U1001\\"}".',
    )
    parser.add_argument("--user-id", default="", help='Caller id for the "agent" command.')
    parser.add_argument("--device-id", default="", help='Device id for the "agent" command.')
    parser.add_argument("--contact", default="", help='Contact for the "agent" command.')
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm the pending action of a stored thread and resume it.",
    )
    parser.add_argument(
        "--cancel",
        action="store_true",
        help="Reject the pending action of a stored thread; no write is performed.",
    )
    parser.add_argument(
        "--thread-id",
        default="",
        help='Conversation id for "chat" and "thread"; one id is one isolated conversation.',
    )
    parser.add_argument(
        "--preference",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Long-term user preference to store, for example --preference contact=email.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Override how many recent messages are rendered into the prompt.",
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


@contextmanager
def _open_answering(
    settings: Settings,
) -> Iterator[tuple[ChunkVectorStore, EmbeddingModel, ChatModel]]:
    """Open retrieval plus the chat model used to write the grounded answer."""

    vectors = ChunkVectorStore(settings.chroma_dir)
    with (
        build_embedding_model(settings) as embedding_model,
        build_chat_model(settings) as chat_model,
    ):
        yield vectors, embedding_model, chat_model


def _answer(
    question: str,
    *,
    top_k: int | None,
    threshold: float | None,
    source: str,
    stream: bool = False,
) -> int:
    """Answer a question strictly from retrieved evidence, with citations."""

    settings = get_settings()
    try:
        with _open_answering(settings) as (vectors, embedding_model, chat_model):
            retriever = Retriever(
                vectors=vectors,
                embedding_model=embedding_model,
                config=RetrieverConfig(
                    top_k=settings.retrieval_top_k,
                    threshold=settings.retrieval_threshold,
                ),
            )
            if stream:
                answer = _stream_grounded_answer(
                    question,
                    retriever=retriever,
                    chat_model=chat_model,
                    top_k=top_k,
                    threshold=threshold,
                    source=source or None,
                )
            else:
                answer = answer_with_context(
                    question,
                    retriever=retriever,
                    chat_model=chat_model,
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

    payload = answer.as_dict()
    status = "ok" if answer.answered else "refused"
    print(
        json.dumps(
            {
                "status": status,
                **{key: value for key, value in payload.items() if key != "answer_status"},
                "answer_status": str(answer.status),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return EXIT_OK if answer.answered else EXIT_PARTIAL_FAILURE


def _stream_grounded_answer(
    question: str,
    *,
    retriever: Retriever,
    chat_model: ChatModel,
    top_k: int | None,
    threshold: float | None,
    source: str | None,
) -> RagAnswer:
    """Print the raw model stream, then return the validated answer.

    Deltas are printed as they arrive so the output is genuinely incremental;
    the structured result is printed by the caller once validation has run.
    """

    prepared = prepare_answer(
        question,
        retriever=retriever,
        top_k=top_k,
        threshold=threshold,
        source=source,
    )
    if prepared.refusal is not None:
        return prepared.refusal

    started = time.perf_counter()
    deltas: list[str] = []
    for delta in stream_raw_answer(prepared, chat_model):
        deltas.append(delta)
        print(delta, end="", flush=True)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print()
    return finalize_answer(
        prepared,
        "".join(deltas),
        model=chat_model.model_name,
        latency_ms=elapsed_ms,
    )


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


def _ingest(path: str, *, force: bool, keep_index_chunks: bool = False) -> int:
    target = Path(path)
    settings = get_settings()
    try:
        with _open_index(settings) as (store, vectors, model):
            if target.is_dir():
                results = ingest_directory(
                    target,
                    store=store,
                    vectors=vectors,
                    embedding_model=model,
                    force=force,
                    drop_index_like_chunks=not keep_index_chunks,
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
                        drop_index_like_chunks=not keep_index_chunks,
                    ),
                )
    except ModelError as exc:
        _error_payload(exc.code, str(exc))
        return EXIT_ERROR
    except IngestionError as exc:
        _error_payload(exc.code, str(exc))
        return EXIT_ERROR

    return _report_results({"target": str(target)}, results)


def _reindex(*, keep_index_chunks: bool = False) -> int:
    settings = get_settings()
    raw_dir = raw_directory(settings.data_dir)
    try:
        with _open_index(settings) as (store, vectors, model):
            results = sync_index(
                raw_dir,
                store=store,
                vectors=vectors,
                embedding_model=model,
                drop_index_like_chunks=not keep_index_chunks,
            )
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


def _evaluate(
    *,
    mode: str,
    dataset: str,
    out: str,
    limit: int | None,
    top_k: int | None,
    threshold: float | None,
    source: str,
) -> int:
    """Run the offline evaluation and write reproducible reports."""

    settings = get_settings()
    dataset_path = Path(dataset) if dataset else settings.data_dir / "eval" / "qa_set.jsonl"
    out_dir = Path(out) if out else settings.data_dir / "eval" / "reports"

    try:
        cases = load_cases(dataset_path)
        if limit is not None:
            cases = cases[:limit]
        if not cases:
            raise EvaluationDatasetError("no cases selected for evaluation")
    except EvaluationDatasetError as exc:
        _error_payload(exc.code, str(exc))
        return EXIT_ERROR

    conditions: dict[str, object] = {
        "dataset": str(dataset_path),
        "case_count": len(cases),
        "top_k": top_k if top_k is not None else settings.retrieval_top_k,
        "threshold": threshold if threshold is not None else settings.retrieval_threshold,
        "source_filter": source or "—",
        "embedding_model": settings.qwen_embedding_model,
        "chunking": {
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        },
        "index_like_filter": True,
        "vector_count": ChunkVectorStore(settings.chroma_dir).count(),
    }

    try:
        if mode == "answer":
            conditions["chat_model"] = settings.qwen_chat_model
            with _open_answering(settings) as (vectors, embedding_model, chat_model):
                retriever = Retriever(
                    vectors=vectors,
                    embedding_model=embedding_model,
                    config=RetrieverConfig(
                        top_k=settings.retrieval_top_k,
                        threshold=settings.retrieval_threshold,
                    ),
                )
                report = run_answer_evaluation(
                    cases,
                    retriever=retriever,
                    chat_model=chat_model,
                    top_k=top_k,
                    threshold=threshold,
                    conditions=conditions,
                )
        else:
            with _open_retrieval(settings) as (vectors, embedding_model):
                retriever = Retriever(
                    vectors=vectors,
                    embedding_model=embedding_model,
                    config=RetrieverConfig(
                        top_k=settings.retrieval_top_k,
                        threshold=settings.retrieval_threshold,
                    ),
                )
                report = run_retrieval_evaluation(
                    cases,
                    retriever=retriever,
                    top_k=top_k,
                    threshold=threshold,
                    conditions=conditions,
                )
    except ModelError as exc:
        _error_payload(exc.code, str(exc))
        return EXIT_ERROR
    except ValueError as exc:
        _error_payload("invalid_argument", str(exc))
        return EXIT_ERROR

    markdown_path, json_path = _write_reports(report, out_dir)
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": report.mode,
                "report_markdown": str(markdown_path),
                "report_json": str(json_path),
                "summary": report.summary.as_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return EXIT_OK


def _write_reports(report: Any, out_dir: Path) -> tuple[Path, Path]:
    """Write the Markdown and JSON reports, returning their paths."""

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    markdown_path = out_dir / f"{report.mode}-{stamp}.md"
    json_path = out_dir / f"{report.mode}-{stamp}.json"
    markdown_path.write_text(to_markdown(report), encoding="utf-8")
    json_path.write_text(to_json(report), encoding="utf-8")
    return markdown_path, json_path


def _tool(argument: str, raw_args: str) -> int:
    """List tool contracts or call one tool without any agent involvement."""

    settings = get_settings()
    name = (argument or "list").strip()

    if name == "list":
        print(
            json.dumps(
                {
                    "status": "ok",
                    "tools": [
                        {
                            "name": tool_name,
                            "args_schema": TOOL_ARG_MODELS[tool_name].model_json_schema(),
                        }
                        for tool_name in TOOL_NAMES
                    ],
                    "note": "所有业务数据均为模拟数据，不代表真实个人或企业信息。",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK

    if name not in TOOL_FUNCTIONS:
        _error_payload(
            "unknown_tool",
            f"unknown tool {name!r}; known tools: {', '.join(TOOL_NAMES)}",
        )
        return EXIT_ERROR

    try:
        payload = json.loads(raw_args) if raw_args.strip() else {}
    except ValueError as exc:
        _error_payload("invalid_argument", f"--args must be a JSON object: {exc}")
        return EXIT_ERROR
    if not isinstance(payload, dict):
        _error_payload("invalid_argument", "--args must be a JSON object")
        return EXIT_ERROR

    try:
        arguments = TOOL_ARG_MODELS[name].model_validate(payload)
    except ValidationError as exc:
        _error_payload("invalid_argument", f"invalid arguments: {exc.error_count()} error(s)")
        return EXIT_ERROR

    seed_path = settings.data_dir / "business" / "seed.json"
    with BusinessRepository(settings.sqlite_path) as repository:
        repository.seed_from_file(seed_path)
        result = TOOL_FUNCTIONS[name](arguments, repository=repository)

    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return EXIT_OK if result.ok else EXIT_PARTIAL_FAILURE


@contextmanager
def _open_agent(
    settings: Settings,
) -> Iterator[tuple[Retriever, ChatModel, BusinessRepository]]:
    """Open everything one agent run needs: retrieval, chat and business data."""

    vectors = ChunkVectorStore(settings.chroma_dir)
    with (
        build_embedding_model(settings) as embedding_model,
        build_chat_model(settings) as chat_model,
        BusinessRepository(settings.sqlite_path) as repository,
    ):
        repository.seed_from_file(settings.data_dir / "business" / "seed.json")
        yield (
            Retriever(vectors=vectors, embedding_model=embedding_model),
            chat_model,
            repository,
        )


def _agent(
    question: str,
    *,
    user_id: str,
    device_id: str,
    contact: str,
) -> int:
    """Run one stateless turn of the LangGraph workflow and print its trace.

    This is the stage 10 view: no checkpointer, so a ticket request stops at
    ``pending_confirmation`` and writes nothing. There is deliberately no
    ``--confirm`` here — resuming a paused run is what ``chat`` does, with a stored
    thread and a real interrupt.
    """

    settings = get_settings()
    try:
        with _open_agent(settings) as (retriever, chat_model, repository):
            run = run_agent(
                question,
                retriever=retriever,
                chat_model=chat_model,
                repository=repository,
                user_id=user_id or None,
                device_id=device_id or None,
                contact=contact or None,
            )
    except ModelError as exc:
        _error_payload(exc.code, str(exc))
        return EXIT_ERROR
    except ValueError as exc:
        _error_payload("invalid_argument", str(exc))
        return EXIT_ERROR

    print(json.dumps(run.as_dict(), ensure_ascii=False, indent=2))
    if run.status in (STATUS_ANSWERED, STATUS_TICKET_CREATED):
        return EXIT_OK
    return EXIT_PARTIAL_FAILURE


@contextmanager
def _open_conversation(
    settings: Settings,
) -> Iterator[tuple[Any, Retriever, ChatModel, BusinessRepository, ConversationStore]]:
    """Open everything a multi-turn conversation needs.

    The checkpointer and the conversation registry share one SQLite file but keep
    separate connections, so a thread listing never depends on the checkpointer
    being open.
    """

    vectors = ChunkVectorStore(settings.chroma_dir)
    with (
        SQLiteCheckpointer(settings.checkpoint_path) as checkpointer,
        ConversationStore(settings.checkpoint_path) as conversations,
        build_embedding_model(settings) as embedding_model,
        build_chat_model(settings) as chat_model,
        BusinessRepository(settings.sqlite_path) as repository,
    ):
        repository.seed_from_file(settings.data_dir / "business" / "seed.json")
        retriever = Retriever(
            vectors=vectors,
            embedding_model=embedding_model,
            config=RetrieverConfig(
                top_k=settings.retrieval_top_k,
                threshold=settings.retrieval_threshold,
            ),
        )
        yield checkpointer, retriever, chat_model, repository, conversations


def _parse_preferences(items: Sequence[str]) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` flags into a mapping."""

    preferences: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"--preference expects KEY=VALUE, got {item!r}")
        preferences[key.strip()] = value.strip()
    return preferences


def _new_thread_id() -> str:
    """A short, sortable, obviously-generated conversation id."""

    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"T{stamp}-{uuid.uuid4().hex[:6]}"


def _chat(
    question: str,
    *,
    thread_id: str,
    user_id: str,
    device_id: str,
    contact: str,
    confirm: bool,
    cancel: bool,
    preferences: Sequence[str],
    window: int | None,
) -> int:
    """Run one turn of a persisted conversation, or resolve a pending action."""

    settings = get_settings()
    thread = thread_id.strip() or _new_thread_id()
    resume = None
    if confirm or cancel:
        resume = {"confirmed": confirm}

    try:
        extra_preferences = _parse_preferences(preferences)
    except ValueError as exc:
        _error_payload("invalid_argument", str(exc))
        return EXIT_ERROR

    try:
        with _open_conversation(settings) as (
            checkpointer,
            retriever,
            chat_model,
            repository,
            conversations,
        ):
            conversations.ensure_thread(thread, user_id or None)
            stored = conversations.load_preferences(user_id or None)
            stored.update(extra_preferences)
            for key, value in extra_preferences.items():
                conversations.save_preference(user_id, key, value)

            graph = build_agent_graph(
                retriever=retriever,
                chat_model=chat_model,
                repository=repository,
                checkpointer=checkpointer,
            )
            turn = chat_turn(
                graph,
                question,
                thread_id=thread,
                user_id=user_id or None,
                device_id=device_id or None,
                contact=contact or None,
                window_size=window if window is not None else settings.conversation_window_size,
                preferences=stored,
                resume=resume,
            )
            conversations.touch_thread(thread)
            _record_turn(graph, thread, turn, user_id=user_id or None, question=question)
    except ThreadOwnershipError as exc:
        _error_payload(exc.code, str(exc))
        return EXIT_ERROR
    except ModelError as exc:
        _error_payload(exc.code, str(exc))
        return EXIT_ERROR
    except ValueError as exc:
        _error_payload("invalid_argument", str(exc))
        return EXIT_ERROR

    print(json.dumps(turn.as_dict(), ensure_ascii=False, indent=2))
    if turn.status in (STATUS_ANSWERED, STATUS_TICKET_CREATED, STATUS_TICKET_CANCELLED):
        return EXIT_OK
    return EXIT_PARTIAL_FAILURE


def _record_turn(
    graph: Any,
    thread_id: str,
    turn: ChatTurn,
    *,
    user_id: str | None,
    question: str,
) -> None:
    """Append this turn to the conversation, which is what makes it multi-turn.

    The append is a state update, not a graph run: it adds the user message and the
    assistant reply to the ``messages`` channel through its reducer, so the next
    turn can read them without replaying any node.
    """

    config = {"configurable": {"thread_id": thread_id}}
    if turn.paused:
        return
    updates: list[dict[str, object]] = [{"role": "user", "content": question}]
    if turn.run.answer:
        updates.append({"role": "assistant", "content": turn.run.answer})
    if not question:
        updates.pop(0)
    if not updates:
        return
    graph.update_state(config, {"messages": updates})


def _thread(action: str, *, thread_id: str, user_id: str, preferences: Sequence[str]) -> int:
    """List, clear or inspect the stored conversations."""

    settings = get_settings()
    name = (action or "list").strip().lower()

    try:
        with ConversationStore(settings.checkpoint_path) as conversations:
            if name == "list":
                threads = conversations.list_threads(user_id or None)
                print(
                    json.dumps(
                        {
                            "status": "ok",
                            "user_id": user_id or None,
                            "thread_count": len(threads),
                            "threads": [thread.as_dict() for thread in threads],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return EXIT_OK

            if name == "clear":
                if not thread_id.strip():
                    _error_payload("invalid_argument", "thread clear requires --thread-id")
                    return EXIT_ERROR
                owner = conversations.owner_of(thread_id)
                if owner is not None and (user_id or None) != owner:
                    _error_payload(
                        "thread_ownership_conflict",
                        f"thread {thread_id!r} belongs to another user",
                    )
                    return EXIT_ERROR
                removed = conversations.clear_thread(thread_id)
                print(
                    json.dumps(
                        {
                            "status": "ok",
                            "thread_id": thread_id,
                            "checkpoints_removed": removed,
                            "note": "长期偏好保存在 user_preferences 中，不随会话清除。",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return EXIT_OK

            if name == "preferences":
                parsed = _parse_preferences(preferences)
                for key, value in parsed.items():
                    if not user_id:
                        _error_payload("invalid_argument", "thread preferences requires --user-id")
                        return EXIT_ERROR
                    conversations.save_preference(user_id, key, value)
                stored = conversations.load_preferences(user_id or None)
                print(
                    json.dumps(
                        {
                            "status": "ok",
                            "user_id": user_id or None,
                            "preferences": stored,
                            "rendered": render_prompt_context(
                                ConversationWindow(messages=()),
                                stored,
                            ),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return EXIT_OK
    except ValueError as exc:
        _error_payload("invalid_argument", str(exc))
        return EXIT_ERROR

    _error_payload(
        "unknown_action", f"unknown thread action {name!r}; use list, clear or preferences"
    )
    return EXIT_ERROR


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

    if args.command == "answer":
        question = (args.argument or "").strip()
        if not question:
            parser.error('answer requires a question, for example: rag-agent answer "主刷卡住"')
        return _answer(
            question,
            top_k=args.top_k,
            threshold=args.threshold,
            source=args.source,
            stream=args.stream,
        )

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
        return _ingest(path, force=args.force, keep_index_chunks=args.keep_index_chunks)

    if args.command == "reindex":
        return _reindex(keep_index_chunks=args.keep_index_chunks)

    if args.command == "delete-document":
        source = (args.argument or "").strip()
        if not source:
            parser.error("delete-document requires the stored source label")
        return _delete_document(source)

    if args.command == "evaluate":
        if args.limit is not None and args.limit <= 0:
            parser.error("--limit must be positive")
        return _evaluate(
            mode=args.mode,
            dataset=args.dataset,
            out=args.out,
            limit=args.limit,
            top_k=args.top_k,
            threshold=args.threshold,
            source=args.source,
        )

    if args.command == "tool":
        return _tool(args.argument or "", args.args)

    if args.command == "agent":
        question = (args.argument or "").strip()
        if not question:
            parser.error(
                'agent requires a question, for example: rag-agent agent "D2001 还在保修吗"'
            )
        if args.confirm or args.cancel:
            parser.error(
                "agent is stateless and never resumes a paused run; use "
                "rag-agent chat --thread-id T1 --user-id U1001 --confirm"
            )
        return _agent(
            question,
            user_id=args.user_id,
            device_id=args.device_id,
            contact=args.contact,
        )

    if args.command == "chat":
        question = (args.argument or "").strip()
        if args.confirm and args.cancel:
            parser.error("--confirm and --cancel cannot be combined")
        if not question and not (args.confirm or args.cancel):
            parser.error(
                "chat requires a question, for example: "
                'rag-agent chat "D2001 还在保修吗" --thread-id T1 --user-id U1001'
            )
        if args.window is not None and args.window <= 0:
            parser.error("--window must be positive")
        return _chat(
            question,
            thread_id=args.thread_id,
            user_id=args.user_id,
            device_id=args.device_id,
            contact=args.contact,
            confirm=args.confirm,
            cancel=args.cancel,
            preferences=args.preference,
            window=args.window,
        )

    if args.command == "thread":
        return _thread(
            args.argument or "list",
            thread_id=args.thread_id,
            user_id=args.user_id,
            preferences=args.preference,
        )

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
