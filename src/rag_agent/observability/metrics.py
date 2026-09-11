"""Runtime metrics for one turn, measured rather than estimated.

The project already had metrics for *offline evaluation* (``evaluation.metrics``). What it
did not have was any record of what happened during a real request: which stage was slow,
how many tokens a turn cost, whether retrieval found anything, and which tools failed.

Every number here is measured at the place the work happens. Nothing is derived from a
timestamp difference taken somewhere convenient, because a stage that is never timed looks
fast for the wrong reason. The collector is a plain mutable object passed down the call
path, and :meth:`TurnMetrics.as_dict` is the only surface other layers read, so the shape
stays stable whatever gets added later.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(slots=True)
class StageTiming:
    """One named step of a turn."""

    name: str
    duration_ms: float = 0.0
    ok: bool = True
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 1),
            "ok": self.ok,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload


@dataclass(slots=True)
class TurnMetrics:
    """Everything measured while answering one turn."""

    thread_id: str = ""
    stages: list[StageTiming] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_calls: int = 0
    retrieval_hits: int = 0
    retrieval_best_score: float | None = None
    retrieval_confident: bool | None = None
    tool_calls: int = 0
    tool_failures: int = 0
    tool_error_codes: list[str] = field(default_factory=list)
    injection_labels: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    # -------------------------------------------------------------- recording

    @contextmanager
    def stage(self, name: str) -> Iterator[StageTiming]:
        """Time one step, recording failure without swallowing it."""

        timing = StageTiming(name=name)
        started = time.perf_counter()
        self.stages.append(timing)
        try:
            yield timing
        except BaseException as exc:
            timing.ok = False
            timing.detail = type(exc).__name__
            raise
        finally:
            timing.duration_ms = (time.perf_counter() - started) * 1000.0

    def record_model_call(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        stage: str = "model",
        duration_ms: float | None = None,
    ) -> None:
        """Record one model call. Token counts stay ``None``-safe for providers that omit them."""

        self.model_calls += 1
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)
        if duration_ms is not None:
            self.stages.append(
                StageTiming(name=stage, duration_ms=duration_ms, detail="model_call")
            )

    def record_retrieval(
        self,
        *,
        hits: int,
        best_score: float | None = None,
        confident: bool | None = None,
        duration_ms: float | None = None,
    ) -> None:
        self.retrieval_hits = hits
        self.retrieval_best_score = best_score
        self.retrieval_confident = confident
        if duration_ms is not None:
            self.stages.append(
                StageTiming(name="retrieval", duration_ms=duration_ms, detail=f"hits={hits}")
            )

    def record_tool_result(self, *, tool: str, ok: bool, error_code: str | None = None) -> None:
        self.tool_calls += 1
        if not ok:
            self.tool_failures += 1
            if error_code:
                self.tool_error_codes.append(error_code)
        self.stages.append(
            StageTiming(
                name=f"tool:{tool}",
                duration_ms=0.0,
                ok=ok,
                detail=error_code or "ok",
            )
        )

    def record_injection_findings(self, labels: tuple[str, ...]) -> None:
        for label in labels:
            if label not in self.injection_labels:
                self.injection_labels.append(label)

    # ---------------------------------------------------------------- reading

    @property
    def total_ms(self) -> float:
        """Wall-clock time for the turn.

        A caller that measured the turn itself records it as the ``turn`` stage, and that
        measurement wins. Otherwise the time since this collector was created is reported —
        which is only right when the collector was built *before* the work started.

        Relying on the elapsed time alone was a real bug: the collector is built after the
        graph returns, so it reported a few milliseconds for a turn that took seconds. The
        fallback is kept because a caller that only builds the collector afterwards would
        otherwise silently report zero.
        """

        for stage in self.stages:
            if stage.name == "turn":
                return stage.duration_ms
        return (time.perf_counter() - self.started_at) * 1000.0

    @property
    def tool_success_rate(self) -> float:
        """Share of tool calls that succeeded; ``1.0`` when nothing was called.

        A turn with no tool calls has not *failed* at anything, so reporting ``0.0`` would be
        a misleading summary line.
        """

        if self.tool_calls == 0:
            return 1.0
        return (self.tool_calls - self.tool_failures) / self.tool_calls

    def as_dict(self) -> dict[str, object]:
        return {
            "thread_id": self.thread_id,
            "total_ms": round(self.total_ms, 1),
            "stages": [stage.as_dict() for stage in self.stages],
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.prompt_tokens + self.completion_tokens,
            },
            "model_calls": self.model_calls,
            "retrieval": {
                "hits": self.retrieval_hits,
                "best_score": (
                    None
                    if self.retrieval_best_score is None
                    else round(self.retrieval_best_score, 4)
                ),
                "confident": self.retrieval_confident,
            },
            "tools": {
                "calls": self.tool_calls,
                "failures": self.tool_failures,
                "success_rate": round(self.tool_success_rate, 4),
                "error_codes": list(self.tool_error_codes),
            },
            "injection_suspected": list(self.injection_labels),
        }


__all__ = ["StageTiming", "TurnMetrics"]
