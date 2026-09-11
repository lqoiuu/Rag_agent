"""Answer generation grounded in retrieved chunks.

Three guarantees are enforced here rather than left to the prompt:

1. **No evidence, no model call.** When retrieval returns nothing, the answer is
   refused without spending a request.
2. **Citations must exist.** Any citation id outside the numbered context is
   dropped and reported, and an answer with no surviving citation is refused.
3. **Unparseable output is refused, not guessed at.** A model reply that is not
   the required JSON never becomes a user-visible answer.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum

from rag_agent.domain.citation import Citation
from rag_agent.domain.retrieval import RetrievalHit, RetrievalResult
from rag_agent.providers.base import ChatMessage, ChatModel, ChatResponse
from rag_agent.retrieval.retriever import Retriever

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_CONTEXT_CHARS = 6000
RAW_OUTPUT_LIMIT = 500

RAG_SYSTEM_PROMPT = (
    "你是企业售后知识库助手。只能依据提供的编号资料回答问题，必须遵守以下规则：\n"
    "1. 只使用资料中明确写到的内容，不要用常识补充，也不要编造。\n"
    "2. 每个事实结论后面用方括号标注依据编号，例如 [1] 或 [1][3]。\n"
    "3. 如果资料不足以回答，把 status 设为 insufficient 并说明缺什么，不要猜测。\n"
    "4. citations 必须是整数数组，写成 [1, 2] 这种形式，禁止写成 [1][2]。\n"
    "5. citations 里的编号是每条资料开头的 [n]，表示第几条资料；"
    "它不是说明书里的章节号、表格序号或页码，不要混用。\n"
    "6. 只输出一个 JSON 对象，不要输出任何解释性文字或 Markdown 代码块。\n"
    'JSON 格式：{"status": "answered" | "insufficient", "answer": "中文回答", '
    '"citations": [资料编号], "reason": "status 为 insufficient 时的原因"}'
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class AnswerStatus(StrEnum):
    """Whether the answer is grounded in evidence or refused."""

    ANSWERED = "answered"
    REFUSED = "refused"


class RefusalCause(StrEnum):
    """Machine-readable reason for a refusal, so refusals can be counted."""

    NO_HITS = "no_hits"
    MODEL_INSUFFICIENT = "model_insufficient"
    UNPARSEABLE = "unparseable"
    UNKNOWN_STATUS = "unknown_status"
    EMPTY_ANSWER = "empty_answer"
    NO_VALID_CITATION = "no_valid_citation"


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """Structured answer plus the evidence and validation results behind it."""

    question: str
    status: AnswerStatus
    text: str
    reason: str
    citations: tuple[Citation, ...] = ()
    hits: tuple[RetrievalHit, ...] = ()
    dropped_citations: tuple[int, ...] = ()
    refusal_cause: RefusalCause | None = None
    confident: bool = False
    model: str | None = None
    latency_ms: float = 0.0
    attempts: int = 0
    raw_output: str = ""

    @property
    def answered(self) -> bool:
        return self.status is AnswerStatus.ANSWERED

    def as_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer_status": str(self.status),
            "refusal_cause": None if self.refusal_cause is None else str(self.refusal_cause),
            "answer": self.text,
            "reason": self.reason,
            "citations": [citation.as_dict() for citation in self.citations],
            "dropped_citations": list(self.dropped_citations),
            "confident": self.confident,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "attempts": self.attempts,
            "evidence": [
                {
                    "rank": hit.rank,
                    "score": round(hit.score, 4),
                    "chunk_id": hit.chunk.chunk_id,
                    "source": hit.chunk.source,
                    "page": hit.chunk.page,
                }
                for hit in self.hits
            ],
            "raw_output": self.raw_output,
        }


@dataclass(frozen=True, slots=True)
class _ModelPayload:
    status: str
    answer: str
    reason: str
    citations: tuple[int, ...]


def format_location(hit: RetrievalHit) -> str:
    """Describe where a hit came from, for the numbered context."""

    parts = [hit.chunk.source]
    if hit.chunk.page is not None:
        parts.append(f"第 {hit.chunk.page} 页")
    if hit.chunk.heading:
        parts.append(hit.chunk.heading)
    return " · ".join(parts)


def build_context(
    hits: Sequence[RetrievalHit], *, max_chars: int = DEFAULT_MAX_CONTEXT_CHARS
) -> tuple[str, tuple[RetrievalHit, ...]]:
    """Render hits as a numbered context within a character budget.

    Returns the context text together with exactly the hits that were included,
    so citation ids can be validated against the same numbering the model saw.
    A single oversized chunk is still included rather than producing no context.
    """

    blocks: list[str] = []
    used_hits: list[RetrievalHit] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        block = f"[{index}] 来源：{format_location(hit)}\n{hit.chunk.content}"
        if used_hits and used + len(block) > max_chars:
            break
        blocks.append(block)
        used_hits.append(hit)
        used += len(block)
    return "\n\n".join(blocks), tuple(used_hits)


def build_user_prompt(question: str, context: str) -> str:
    return f"资料：\n{context}\n\n问题：{question}"


@dataclass(frozen=True, slots=True)
class PreparedAnswer:
    """Retrieval result plus the numbered context, ready for one model call.

    ``refusal`` is set when there is no evidence at all, which is the case that
    must never reach the model.
    """

    question: str
    retrieval: RetrievalResult
    context: str
    hits: tuple[RetrievalHit, ...]
    refusal: RagAnswer | None = None

    @property
    def has_evidence(self) -> bool:
        return self.refusal is None


def prepare_answer(
    question: str,
    *,
    retriever: Retriever,
    top_k: int | None = None,
    threshold: float | None = None,
    source: str | None = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> PreparedAnswer:
    """Retrieve evidence and build the numbered context."""

    retrieval = retriever.search(question, top_k=top_k, threshold=threshold, source=source)
    if not retrieval.hits:
        LOGGER.info("answer refused question_chars=%d cause=no_hits", len(question))
        return PreparedAnswer(
            question=question,
            retrieval=retrieval,
            context="",
            hits=(),
            refusal=RagAnswer(
                question=question,
                status=AnswerStatus.REFUSED,
                text="",
                reason="知识库中没有检索到任何相关资料，无法回答该问题。",
                refusal_cause=RefusalCause.NO_HITS,
                confident=False,
            ),
        )

    context, used_hits = build_context(retrieval.hits, max_chars=max_context_chars)
    return PreparedAnswer(
        question=question,
        retrieval=retrieval,
        context=context,
        hits=used_hits,
    )


def build_messages(prepared: PreparedAnswer) -> list[ChatMessage]:
    """Build the grounded prompt for a prepared answer."""

    return [
        ChatMessage(role="system", content=RAG_SYSTEM_PROMPT),
        ChatMessage(role="user", content=build_user_prompt(prepared.question, prepared.context)),
    ]


def answer_with_context(
    question: str,
    *,
    retriever: Retriever,
    chat_model: ChatModel,
    top_k: int | None = None,
    threshold: float | None = None,
    source: str | None = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    temperature: float = 0.0,
) -> RagAnswer:
    """Retrieve evidence, then answer strictly from it."""

    prepared = prepare_answer(
        question,
        retriever=retriever,
        top_k=top_k,
        threshold=threshold,
        source=source,
        max_context_chars=max_context_chars,
    )
    if prepared.refusal is not None:
        return prepared.refusal

    response = chat_model.chat(build_messages(prepared), temperature=temperature)
    return finalize_answer(
        prepared,
        response.text,
        model=response.model,
        latency_ms=response.latency_ms,
        attempts=response.attempts,
    )


def stream_raw_answer(
    prepared: PreparedAnswer,
    chat_model: ChatModel,
    *,
    temperature: float = 0.0,
) -> Iterator[str]:
    """Yield the raw model output for a prepared answer, delta by delta.

    The text is intentionally *raw*: the grounded answer format is JSON, so the
    caller accumulates the deltas and passes them to :func:`finalize_answer`
    afterwards instead of showing unvalidated content.
    """

    yield from chat_model.stream_chat(build_messages(prepared), temperature=temperature)


def finalize_answer(
    prepared: PreparedAnswer,
    raw_text: str,
    *,
    model: str | None,
    latency_ms: float = 0.0,
    attempts: int = 1,
) -> RagAnswer:
    """Validate a raw model reply against the prepared context."""

    question = prepared.question
    used_hits = prepared.hits
    confident = prepared.retrieval.is_confident
    response = ChatResponse(
        text=raw_text, model=model or "", latency_ms=latency_ms, attempts=attempts
    )

    payload = parse_model_payload(raw_text)
    if payload is None:
        LOGGER.warning("answer refused question_chars=%d cause=unparseable", len(question))
        return RagAnswer(
            question=question,
            status=AnswerStatus.REFUSED,
            text="",
            reason="模型没有返回约定的 JSON，答案已被拒绝，请重试或改问更具体的问题。",
            hits=used_hits,
            refusal_cause=RefusalCause.UNPARSEABLE,
            confident=confident,
            model=response.model,
            latency_ms=response.latency_ms,
            attempts=response.attempts,
            raw_output=response.text[:RAW_OUTPUT_LIMIT],
        )

    valid_ids = sorted({value for value in payload.citations if 1 <= value <= len(used_hits)})
    dropped = tuple(value for value in payload.citations if value not in valid_ids)

    if payload.status == "insufficient":
        return RagAnswer(
            question=question,
            status=AnswerStatus.REFUSED,
            text="",
            reason=payload.reason or "资料不足以回答该问题。",
            hits=used_hits,
            dropped_citations=dropped,
            refusal_cause=RefusalCause.MODEL_INSUFFICIENT,
            confident=confident,
            model=response.model,
            latency_ms=response.latency_ms,
            attempts=response.attempts,
            raw_output=response.text[:RAW_OUTPUT_LIMIT],
        )

    if payload.status != "answered":
        return _refused(
            question,
            RefusalCause.UNKNOWN_STATUS,
            f"模型返回了无法识别的状态 {payload.status!r}。",
            confident,
            used_hits,
            dropped,
            response,
        )
    if not payload.answer:
        return _refused(
            question,
            RefusalCause.EMPTY_ANSWER,
            "模型没有给出回答正文。",
            confident,
            used_hits,
            dropped,
            response,
        )
    if not valid_ids:
        return _refused(
            question,
            RefusalCause.NO_VALID_CITATION,
            "回答没有可核验的引用编号，已拒绝输出以免给出无依据的结论。",
            confident,
            used_hits,
            dropped,
            response,
        )

    citations = tuple(
        Citation.from_chunk(citation_id, used_hits[citation_id - 1].chunk)
        for citation_id in valid_ids
    )
    LOGGER.info(
        "answer produced question_chars=%d citations=%d dropped=%d",
        len(question),
        len(citations),
        len(dropped),
    )
    return RagAnswer(
        question=question,
        status=AnswerStatus.ANSWERED,
        text=payload.answer,
        reason=payload.reason or f"依据 {len(citations)} 个资料分片回答。",
        citations=citations,
        hits=used_hits,
        dropped_citations=dropped,
        confident=confident,
        model=response.model,
        latency_ms=response.latency_ms,
        attempts=response.attempts,
        raw_output=response.text[:RAW_OUTPUT_LIMIT],
    )


def parse_model_payload(text: str) -> _ModelPayload | None:
    """Extract the required JSON object from a model reply."""

    match = _JSON_OBJECT.search(text)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None

    return _ModelPayload(
        status=str(payload.get("status", "")).strip().lower(),
        answer=str(payload.get("answer", "")).strip(),
        reason=str(payload.get("reason", "")).strip(),
        citations=tuple(_as_int(value) for value in _as_list(payload.get("citations"))),
    )


def _refused(
    question: str,
    cause: RefusalCause,
    reason: str,
    confident: bool,
    hits: tuple[RetrievalHit, ...],
    dropped: tuple[int, ...],
    response: ChatResponse,
) -> RagAnswer:
    LOGGER.warning("answer refused question_chars=%d cause=%s", len(question), cause)
    return RagAnswer(
        question=question,
        status=AnswerStatus.REFUSED,
        text="",
        reason=reason,
        hits=hits,
        dropped_citations=dropped,
        refusal_cause=cause,
        confident=confident,
        model=response.model,
        latency_ms=response.latency_ms,
        attempts=response.attempts,
        raw_output=str(response.text)[:RAW_OUTPUT_LIMIT],
    )


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return 0
