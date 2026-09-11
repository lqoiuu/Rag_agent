"""Logging, metrics, trust boundaries and failure policy."""

from rag_agent.observability.degradation import (
    MODEL_PLANS,
    RETRIEVAL_PLANS,
    TOOL_PLANS,
    DegradationPlan,
    DegradeAction,
    for_model_error,
    for_retrieval_outcome,
    for_tool_error,
)
from rag_agent.observability.logging import configure_logging
from rag_agent.observability.metrics import StageTiming, TurnMetrics
from rag_agent.observability.untrusted import (
    EVIDENCE_END,
    EVIDENCE_RULE,
    EVIDENCE_START,
    InjectionFinding,
    injection_labels,
    scan_injection,
    wrap_untrusted,
)

__all__ = [
    "EVIDENCE_END",
    "EVIDENCE_RULE",
    "EVIDENCE_START",
    "MODEL_PLANS",
    "RETRIEVAL_PLANS",
    "TOOL_PLANS",
    "DegradationPlan",
    "DegradeAction",
    "InjectionFinding",
    "StageTiming",
    "TurnMetrics",
    "configure_logging",
    "for_model_error",
    "for_retrieval_outcome",
    "for_tool_error",
    "injection_labels",
    "scan_injection",
    "wrap_untrusted",
]
