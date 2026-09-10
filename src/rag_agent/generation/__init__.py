"""Answer generation built on top of the provider protocols."""

from rag_agent.generation.minimal_qa import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    MinimalAnswer,
    answer_question,
    build_minimal_qa_chain,
    to_chat_messages,
)

__all__ = [
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "MinimalAnswer",
    "answer_question",
    "build_minimal_qa_chain",
    "to_chat_messages",
]
