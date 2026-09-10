"""Minimal ``Prompt -> Model -> Parser`` chain assembled with LCEL.

This is the smallest end-to-end path through the model layer: no retrieval, no
tools, no graph. It exists to prove the provider protocol, the prompt rendering,
and the structured parser work together before later stages add RAG.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from rag_agent.providers.base import ChatMessage, ChatModel, ChatResponse, ChatRole

SYSTEM_PROMPT = (
    "你是一个最小问答链路，只用于验证模型接入层是否可用。"
    "请用简洁的中文直接回答问题；如果不确定，就明确说明不确定，不要编造事实。"
)
USER_TEMPLATE = "{question}"

_ROLE_BY_MESSAGE_TYPE: dict[str, ChatRole] = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
}


class MinimalAnswer(BaseModel):
    """Structured chain output that keeps the model evidence attached.

    Returning a model instead of a bare string is deliberate: ``model``,
    ``latency_ms`` and ``attempts`` are the raw material for the evaluation and
    observability stages, so they must not be dropped here.
    """

    text: str
    model: str
    latency_ms: float
    attempts: int


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def to_chat_messages(prompt_value: PromptValue) -> list[ChatMessage]:
    """Convert an LCEL prompt value into provider-neutral messages."""

    messages: list[ChatMessage] = []
    for message in prompt_value.to_messages():
        role = _ROLE_BY_MESSAGE_TYPE.get(message.type)
        if role is None:
            raise ValueError(f"unsupported prompt message type: {message.type}")
        messages.append(ChatMessage(role=role, content=_message_text(message.content)))
    return messages


def _to_answer(response: ChatResponse) -> MinimalAnswer:
    return MinimalAnswer(
        text=response.text,
        model=response.model,
        latency_ms=response.latency_ms,
        attempts=response.attempts,
    )


def build_minimal_qa_chain(model: ChatModel) -> Runnable[dict[str, Any], MinimalAnswer]:
    """Build the reusable ``Prompt -> Model -> Parser`` runnable.

    The chain depends on the ``ChatModel`` protocol only, so a fake, a real
    provider, or a future vendor adapter can all be dropped in unchanged.
    """

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_TEMPLATE)])
    to_messages = RunnableLambda[PromptValue, list[ChatMessage]](to_chat_messages)
    call_model = RunnableLambda[list[ChatMessage], ChatResponse](
        lambda messages: model.chat(messages)
    )
    to_answer = RunnableLambda[ChatResponse, MinimalAnswer](_to_answer)
    return prompt | to_messages | call_model | to_answer


def answer_question(model: ChatModel, question: str) -> MinimalAnswer:
    """Run the minimal chain once for a single question."""

    return build_minimal_qa_chain(model).invoke({"question": question})
