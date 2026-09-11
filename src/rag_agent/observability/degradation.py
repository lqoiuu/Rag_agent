"""Degradation policy: what the system does when a dependency fails.

The failure codes already existed and were already classified (``providers.base.ModelError``
carries ``retryable``, ``tools.errors.ToolError`` carries a stable code, ingestion errors have
their own). What did not exist was a single place answering the question that matters to the
caller: *given this failure, do I retry, answer partially, refuse, or report an error — and
what do I tell the user?*

This module is that table. It deliberately maps codes to a small set of **actions** rather
than to free text, so the behaviour is testable and cannot drift into "some component
swallows the error and returns a friendly sentence that looks like an answer".

The one rule worth stating out loud: no action here ever produces a factual answer. A
degraded turn either says what it could not do, or returns nothing with a classified error.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DegradeAction(StrEnum):
    """What the caller should do about a failure."""

    RETRY = "retry"
    """Transient: the same call may succeed without changing anything."""

    ANSWER_PARTIAL = "answer_partial"
    """Answer from what is already available, and say what was skipped."""

    REFUSE = "refuse"
    """Cannot answer at all; the refusal must name the reason."""

    REPORT_ERROR = "report_error"
    """The request could not be attempted or the result is untrustworthy."""


@dataclass(frozen=True, slots=True)
class DegradationPlan:
    """One row of the policy table."""

    action: DegradeAction
    retryable: bool
    user_message: str
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "action": str(self.action),
            "retryable": self.retryable,
            "user_message": self.user_message,
        }


#: Model failures. Timeouts and rate limits are transient; auth and request errors are not,
#: and retrying them only spends money.
MODEL_PLANS: dict[str, DegradationPlan] = {
    "timeout": DegradationPlan(
        action=DegradeAction.RETRY,
        retryable=True,
        user_message="模型响应超时，请稍后重试。",
    ),
    "connection": DegradationPlan(
        action=DegradeAction.RETRY,
        retryable=True,
        user_message="无法连接模型服务，请检查网络后重试。",
    ),
    "rate_limit": DegradationPlan(
        action=DegradeAction.RETRY,
        retryable=True,
        user_message="模型服务限流，请稍后重试。",
    ),
    "server": DegradationPlan(
        action=DegradeAction.RETRY,
        retryable=True,
        user_message="模型服务暂时不可用，请稍后重试。",
    ),
    "auth": DegradationPlan(
        action=DegradeAction.REPORT_ERROR,
        retryable=False,
        user_message="模型服务鉴权失败：请检查本地 .env 中的密钥配置。",
    ),
    "request": DegradationPlan(
        action=DegradeAction.REPORT_ERROR,
        retryable=False,
        user_message="请求被模型服务拒绝，属于配置或参数问题，重试无效。",
    ),
    "response": DegradationPlan(
        action=DegradeAction.REPORT_ERROR,
        retryable=False,
        user_message="模型返回了无法解析的内容，本次回答不可信，已丢弃。",
    ),
}

#: Business tool failures.
TOOL_PLANS: dict[str, DegradationPlan] = {
    "not_found": DegradationPlan(
        action=DegradeAction.ANSWER_PARTIAL,
        retryable=False,
        user_message="没有查到对应记录，请确认编号是否正确。",
    ),
    "permission_denied": DegradationPlan(
        action=DegradeAction.ANSWER_PARTIAL,
        retryable=False,
        user_message="没有权限访问该记录：只能查询你自己的设备与订单。",
    ),
    "invalid_argument": DegradationPlan(
        action=DegradeAction.ANSWER_PARTIAL,
        retryable=False,
        user_message="请求参数不合法，请补充或修正后重试。",
    ),
    "confirmation_required": DegradationPlan(
        action=DegradeAction.ANSWER_PARTIAL,
        retryable=False,
        user_message="该操作需要你先确认才会执行。",
    ),
    "conflict": DegradationPlan(
        action=DegradeAction.REPORT_ERROR,
        retryable=False,
        user_message="数据状态冲突，重试无法解决，需要人工处理。",
    ),
    "unavailable": DegradationPlan(
        action=DegradeAction.RETRY,
        retryable=True,
        user_message="业务数据存储暂时不可用，请稍后重试。",
    ),
}

#: Retrieval and generation outcomes that are not exceptions but still change the answer.
RETRIEVAL_PLANS: dict[str, DegradationPlan] = {
    "no_hits": DegradationPlan(
        action=DegradeAction.REFUSE,
        retryable=False,
        user_message="知识库里没有检索到相关资料，无法回答这个问题。",
    ),
    "model_insufficient": DegradationPlan(
        action=DegradeAction.REFUSE,
        retryable=False,
        user_message="资料中没有足够信息回答这个问题。",
    ),
    "no_valid_citation": DegradationPlan(
        action=DegradeAction.REFUSE,
        retryable=False,
        user_message="回答没有可核验的来源，已按拒答处理。",
    ),
    "unparseable": DegradationPlan(
        action=DegradeAction.REFUSE,
        retryable=True,
        user_message="模型输出格式异常，本次回答已丢弃，可以重试。",
    ),
    "unknown_status": DegradationPlan(
        action=DegradeAction.REFUSE,
        retryable=True,
        user_message="模型返回了无法识别的状态，本次回答已丢弃。",
    ),
    "empty_answer": DegradationPlan(
        action=DegradeAction.REFUSE,
        retryable=True,
        user_message="模型没有给出有效内容，可以重试。",
    ),
    "index_unavailable": DegradationPlan(
        action=DegradeAction.REPORT_ERROR,
        retryable=False,
        user_message="向量索引不可用，检索无法进行；请检查索引目录或重新入库。",
    ),
}

_UNKNOWN = DegradationPlan(
    action=DegradeAction.REPORT_ERROR,
    retryable=False,
    user_message="发生了未分类的错误，已停止本次处理。",
)


def _lookup(code: str | None, table: dict[str, DegradationPlan]) -> DegradationPlan:
    if not code:
        return _UNKNOWN
    return table.get(code, _UNKNOWN)


def for_model_error(code: str | None) -> DegradationPlan:
    """Policy for a provider-layer failure."""

    return _lookup(code, MODEL_PLANS)


def for_tool_error(code: str | None) -> DegradationPlan:
    """Policy for a business-tool failure."""

    return _lookup(code, TOOL_PLANS)


def for_retrieval_outcome(code: str | None) -> DegradationPlan:
    """Policy for a refusal or retrieval outcome.

    Name matters: this table answers "why was no answer produced", which is a different
    question from "what failed". Both end up in the same place in the UI, and both must say
    which of the two happened.
    """

    return _lookup(code, RETRIEVAL_PLANS)


__all__ = [
    "MODEL_PLANS",
    "RETRIEVAL_PLANS",
    "TOOL_PLANS",
    "DegradationPlan",
    "DegradeAction",
    "for_model_error",
    "for_retrieval_outcome",
    "for_tool_error",
]
