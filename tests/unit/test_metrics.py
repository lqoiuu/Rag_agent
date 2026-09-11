"""Runtime metrics: measured at the stage, and honest about what was not measured."""

from __future__ import annotations

import time

from rag_agent.observability import TurnMetrics


def test_a_timed_stage_reports_its_own_duration() -> None:
    metrics = TurnMetrics(thread_id="T1")

    with metrics.stage("classify"):
        time.sleep(0.01)

    assert metrics.stages[0].name == "classify"
    assert metrics.stages[0].ok is True
    assert metrics.stages[0].duration_ms >= 10.0


def test_a_failed_stage_is_recorded_and_the_error_still_propagates() -> None:
    """Swallowing the error to keep the timing would hide the only interesting case."""

    metrics = TurnMetrics()

    try:
        with metrics.stage("retrieval"):
            raise RuntimeError("store is down")
    except RuntimeError:
        pass
    else:  # pragma: no cover - 断言用
        raise AssertionError("异常应当继续抛出")

    assert metrics.stages[0].ok is False
    assert metrics.stages[0].detail == "RuntimeError"


def test_model_calls_accumulate_tokens_and_survive_missing_usage() -> None:
    """A provider may omit token counts; that must add zero, not raise."""

    metrics = TurnMetrics()

    metrics.record_model_call(prompt_tokens=120, completion_tokens=40, duration_ms=8.5)
    metrics.record_model_call(prompt_tokens=None, completion_tokens=None, duration_ms=3.0)

    assert metrics.model_calls == 2
    assert metrics.prompt_tokens == 120
    assert metrics.completion_tokens == 40
    assert [stage.name for stage in metrics.stages] == ["model", "model"]


def test_retrieval_records_hits_score_and_confidence() -> None:
    metrics = TurnMetrics()

    metrics.record_retrieval(hits=5, best_score=0.5889, confident=True, duration_ms=61.0)

    assert metrics.retrieval_hits == 5
    assert metrics.retrieval_best_score == 0.5889
    assert metrics.retrieval_confident is True
    payload = metrics.as_dict()
    assert payload["retrieval"] == {"hits": 5, "best_score": 0.5889, "confident": True}


def test_tool_results_count_failures_and_collect_error_codes() -> None:
    metrics = TurnMetrics()

    metrics.record_tool_result(tool="device.lookup", ok=True)
    metrics.record_tool_result(tool="order.lookup", ok=False, error_code="permission_denied")

    assert metrics.tool_calls == 2
    assert metrics.tool_failures == 1
    assert metrics.tool_error_codes == ["permission_denied"]
    assert metrics.tool_success_rate == 0.5


def test_success_rate_without_tool_calls_is_one_not_zero() -> None:
    """A turn that called nothing has not failed at anything."""

    metrics = TurnMetrics()

    assert metrics.tool_calls == 0
    assert metrics.tool_success_rate == 1.0
    assert metrics.as_dict()["tools"] == {
        "calls": 0,
        "failures": 0,
        "success_rate": 1.0,
        "error_codes": [],
    }


def test_injection_labels_are_deduplicated() -> None:
    metrics = TurnMetrics()

    metrics.record_injection_findings(("override_instructions",))
    metrics.record_injection_findings(("override_instructions", "role_hijack"))

    assert metrics.injection_labels == ["override_instructions", "role_hijack"]


def test_as_dict_reports_the_stable_shape() -> None:
    metrics = TurnMetrics(thread_id="T9")

    payload = metrics.as_dict()

    assert payload["thread_id"] == "T9"
    assert set(payload) == {
        "thread_id",
        "total_ms",
        "stages",
        "tokens",
        "model_calls",
        "retrieval",
        "tools",
        "injection_suspected",
    }
    assert payload["tokens"] == {"prompt": 0, "completion": 0, "total": 0}
    assert payload["injection_suspected"] == []
