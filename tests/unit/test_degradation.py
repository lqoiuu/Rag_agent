"""The degradation policy: every failure path, and the promise that none invents an answer."""

from __future__ import annotations

import pytest

from rag_agent.observability import (
    MODEL_PLANS,
    RETRIEVAL_PLANS,
    TOOL_PLANS,
    DegradeAction,
    for_model_error,
    for_retrieval_outcome,
    for_tool_error,
)
from rag_agent.providers.base import (
    ModelAuthError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelRequestError,
    ModelResponseError,
    ModelServerError,
    ModelTimeoutError,
)
from rag_agent.tools.errors import (
    ConfirmationRequiredError,
    ConflictError,
    InvalidArgumentError,
    NotFoundError,
    PermissionDeniedError,
    ToolError,
    ToolUnavailableError,
)

MODEL_ERROR_TYPES = (
    ModelTimeoutError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelServerError,
    ModelAuthError,
    ModelRequestError,
    ModelResponseError,
)

TOOL_ERROR_TYPES = (
    NotFoundError,
    PermissionDeniedError,
    InvalidArgumentError,
    ConfirmationRequiredError,
    ConflictError,
    ToolUnavailableError,
)


def test_every_model_error_code_has_a_plan() -> None:
    """A missing row would silently become ``REPORT_ERROR`` with a generic sentence."""

    for error_type in MODEL_ERROR_TYPES:
        assert error_type.code in MODEL_PLANS, error_type.code


def test_every_tool_error_code_has_a_plan() -> None:
    for error_type in TOOL_ERROR_TYPES:
        assert error_type.code in TOOL_PLANS, error_type.code


def test_transient_failures_are_retryable_and_semantic_ones_are_not() -> None:
    assert for_model_error("timeout").retryable is True
    assert for_model_error("rate_limit").action is DegradeAction.RETRY
    # 鉴权与请求错误重试一万次结果都一样，还会白花钱
    assert for_model_error("auth").retryable is False
    assert for_model_error("request").action is DegradeAction.REPORT_ERROR

    assert for_tool_error("unavailable").retryable is True
    assert for_tool_error("conflict").retryable is False


def test_a_denied_lookup_degrades_to_a_partial_answer_not_a_crash() -> None:
    """A permission failure is a normal outcome the agent reports, not a failed request."""

    plan = for_tool_error("permission_denied")

    assert plan.action is DegradeAction.ANSWER_PARTIAL
    assert "权限" in plan.user_message


def test_a_pending_confirmation_is_not_treated_as_an_error() -> None:
    plan = for_tool_error("confirmation_required")

    assert plan.action is DegradeAction.ANSWER_PARTIAL
    assert plan.retryable is False


@pytest.mark.parametrize(
    "code",
    ["no_hits", "model_insufficient", "no_valid_citation", "unparseable", "unknown_status"],
)
def test_refusals_have_a_reason_and_never_retry_silently(code: str) -> None:
    plan = for_retrieval_outcome(code)

    assert plan.action is DegradeAction.REFUSE
    assert plan.user_message


def test_an_unavailable_index_is_reported_rather_than_refused() -> None:
    """A broken index is not the same thing as "the documents do not cover it"."""

    plan = for_retrieval_outcome("index_unavailable")

    assert plan.action is DegradeAction.REPORT_ERROR
    assert "索引" in plan.user_message


@pytest.mark.parametrize("lookup", [for_model_error, for_tool_error, for_retrieval_outcome])
def test_an_unknown_code_never_becomes_success(lookup: object) -> None:
    plan = lookup("something_nobody_mapped")  # type: ignore[operator]

    assert plan.action is DegradeAction.REPORT_ERROR
    assert plan.retryable is False


@pytest.mark.parametrize("lookup", [for_model_error, for_tool_error, for_retrieval_outcome])
def test_a_missing_code_is_treated_as_unknown(lookup: object) -> None:
    assert lookup(None).action is DegradeAction.REPORT_ERROR  # type: ignore[operator]


def test_no_plan_claims_a_factual_answer() -> None:
    """The property that matters: degrading never produces an answer out of nothing.

    If some future row said "answer anyway", the system would be inventing facts exactly
    when it has least information. The allowed actions are bounded for that reason.
    """

    allowed = {
        DegradeAction.RETRY,
        DegradeAction.ANSWER_PARTIAL,
        DegradeAction.REFUSE,
        DegradeAction.REPORT_ERROR,
    }
    for table in (MODEL_PLANS, TOOL_PLANS, RETRIEVAL_PLANS):
        for plan in table.values():
            assert plan.action in allowed
            assert plan.user_message.strip()


def test_plans_render_for_a_caller() -> None:
    assert for_model_error("timeout").as_dict() == {
        "action": "retry",
        "retryable": True,
        "user_message": "模型响应超时，请稍后重试。",
    }


def test_a_tool_error_carries_a_code_the_policy_recognises() -> None:
    """The two halves must agree: an error raised by a tool must have a plan."""

    error = ToolUnavailableError("store is down")

    assert isinstance(error, ToolError)
    assert for_tool_error(error.code).action is DegradeAction.RETRY
