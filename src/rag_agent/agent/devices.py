"""Device-label matching, shared by the node that looks devices up and the router.

Kept in its own module so that the pattern and the resolution order have exactly one
definition: the node uses it to decide *what to look up*, and the router uses it to
decide *whether a model classification should be corrected*. Two copies of this rule
would eventually disagree, and the failure would look like a random routing bug.
"""

from __future__ import annotations

import re

from rag_agent.agent.state import AgentState

#: A device label such as ``D2002``. Deliberately strict: an uppercase ``D`` followed
#: by digits only. The simulated ids are ``D2001`` to ``D2005``, and no model name in
#: the knowledge base has this shape, so a false positive would have to be invented by
#: the user.
DEVICE_ID_PATTERN = re.compile(r"\bD\d{3,}\b")


def resolve_device_id(state: AgentState) -> str | None:
    """Find the device this turn is about, without asking the model.

    The order is the caller's explicit flag first, then the current message, then the
    conversation window. The window is what makes a follow-up such as
    "那它的保修期是多久" resolvable at all.

    Ownership is deliberately *not* checked here. The extracted id is handed to the
    tool exactly like a caller-supplied one, so a device belonging to somebody else
    still fails with ``permission_denied`` instead of being silently skipped.
    """

    supplied = (state.get("device_id") or "").strip()
    if supplied:
        return supplied
    for text in (state.get("question", ""), state.get("prompt_context", "")):
        match = DEVICE_ID_PATTERN.search(text or "")
        if match:
            return match.group(0).upper()
    return None


__all__ = ["DEVICE_ID_PATTERN", "resolve_device_id"]
