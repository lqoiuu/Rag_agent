"""Conversation page: streaming answers, citations, tool calls and the confirm gate.

Business logic stays out of this file. It calls the same ``chat_turn`` /
``stream_chat_turn`` entry points the CLI uses, so the confirmation pause, the intent
routing and the citation checks behave identically here.

One distinction is load-bearing and visible in the UI: the message list below is what
*this browser session has displayed* (``st.session_state``, lost on refresh), while the
agent's memory is the SQLite checkpoint addressed by the thread id (survives a restart).
The sidebar reports both.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from rag_agent.agent import (
    STATUS_ANSWERED,
    STATUS_PENDING_CONFIRMATION,
    STATUS_TICKET_CANCELLED,
    STATUS_TICKET_CREATED,
    build_agent_graph,
    chat_turn,
    stream_chat_turn,
)
from rag_agent.ui.services import get_resources, session_permissions

STATUS_LABELS = {
    STATUS_ANSWERED: ("已作答", "green"),
    STATUS_PENDING_CONFIRMATION: ("等待确认", "orange"),
    STATUS_TICKET_CREATED: ("工单已创建", "green"),
    STATUS_TICKET_CANCELLED: ("已取消", "gray"),
    "refused": ("已拒答", "red"),
    "needs_input": ("需要补充信息", "orange"),
    "error": ("出错", "red"),
}


def render_citations(citations: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
    """Render the numbered evidence behind an answer."""

    if not citations:
        return
    st.markdown("**引用来源**")
    for citation in citations:
        label = citation.get("label") or citation.get("source") or "未知来源"
        page = citation.get("page")
        score_note = f"第 {page} 页" if page else "无页码"
        with st.container(border=True):
            st.markdown(f"**[{citation.get('citation_id')}] {label}**")
            st.badge(score_note, icon=":material/description:", color="blue")
            excerpt = str(citation.get("excerpt") or "").strip()
            if excerpt:
                st.caption(excerpt)


def render_tool_results(results: tuple[dict[str, Any], ...]) -> None:
    """Show what the business tools returned, one step per call."""

    for entry in results:
        tool = str(entry.get("tool", "tool"))
        ok = entry.get("status") == "ok"
        with st.status(
            f"{tool} · {'成功' if ok else entry.get('error_code')}",
            type="step",
            state="complete" if ok else "error",
        ):
            st.json(entry.get("data") if ok else {"error": entry.get("error_message")})


def render_metrics(turn: Any) -> None:
    """Show what this turn measured, and be explicit about the number that is missing."""

    metrics = turn.metrics
    if metrics is None:
        return
    payload = metrics.as_dict()
    with st.expander("本轮指标（实测）", expanded=False):
        left, middle, right = st.columns(3)
        left.metric("本轮耗时", f"{payload['total_ms']} ms")
        middle.metric("检索命中", payload["retrieval"]["hits"])
        right.metric("工具成功率", f"{payload['tools']['success_rate']:.0%}")
        st.caption(
            "token 用量由 Provider 层记录在日志里，没有进入图状态，因此这里显示为 0 而不是估算值。"
            f"模型调用次数：{payload['model_calls']}。"
        )
        st.json(payload)


def render_turn(turn: Any, *, show_history: bool) -> None:
    """Render one turn: this turn's activity, then its answer, then citations."""

    if turn.run.intent_source == "delegated":
        return

    if turn.run.injection_suspected:
        st.warning(
            "检索到的资料里出现了形似指令的文字（"
            + "、".join(turn.run.injection_suspected)
            + "），已按**数据**处理、未照做。回答仍只依据资料编号，并经过引用校验。",
            icon=":material/gpp_maybe:",
        )

    label, colour = STATUS_LABELS.get(turn.status, (turn.status, "gray"))
    st.badge(label, color=colour, icon=":material/label:")
    st.caption(f"意图：`{turn.run.intent}` · 会话窗口：{len(turn.window.messages)} 条消息")

    if turn.run.error_message:
        st.warning(turn.run.error_message, icon=":material/info:")

    if turn.run.answer:
        st.markdown(turn.run.answer)

    render_tool_results(turn.turn_tool_results)
    render_citations(list(turn.run.citations))
    render_metrics(turn)

    if show_history:
        with st.expander("本轮回合的节点轨迹", expanded=False):
            st.code("\n".join(turn.turn_trace) or "(无)", language="text")
        with st.expander("整条会话累积的轨迹（Checkpoint 中的全量）", expanded=False):
            st.code("\n".join(turn.run.trace) or "(无)", language="text")
        with st.expander("本轮实际使用的 Prompt 上下文", expanded=False):
            window = turn.window
            st.caption(
                f"窗口内消息数：{len(window.messages)}；长期偏好：{len(turn.preferences)} 条"
            )
            for message in window.messages:
                st.markdown(f"- **{message.role}**：{message.content}")


def stream_answer(question: str) -> tuple[Any, str]:
    """Stream one knowledge turn, then return the validated turn and the raw stream.

    ``stream_chat_turn`` yields raw model output (the grounded answer is JSON) followed
    by the validated turn. The raw deltas are therefore used as a *progress signal*, not
    as the answer: the text shown to the user is ``turn.run.answer``, which has passed
    the citation checks. That is what keeps streaming from bypassing validation.
    """

    resources = get_resources()
    graph = build_agent_graph(
        retriever=resources.retriever,
        chat_model=resources.chat_model,
        repository=resources.repository,
        checkpointer=resources.checkpointer,
        permissions=session_permissions(),
    )
    turn: Any = None
    raw_chunks: list[str] = []
    with st.status(":shimmer[正在检索并生成…]", type="compact") as status:
        for delta, candidate in stream_chat_turn(
            graph,
            question,
            retriever=resources.retriever,
            chat_model=resources.chat_model,
            thread_id=st.session_state.thread_id,
            window_size=resources.settings.conversation_window_size,
            preferences=resources.conversations.load_preferences(st.session_state.user_id or None),
            delegate_to_agent=True,
        ):
            if candidate is not None:
                turn = candidate
                continue
            raw_chunks.append(delta)
            status.update(label=f":shimmer[正在生成… 已收到 {len(''.join(raw_chunks))} 字符]")
        status.update(label="生成完成", state="complete")
    return turn, "".join(raw_chunks)


def run_agent_turn(question: str) -> Any:
    """Run one full-agent turn (device lookups and the ticket flow included)."""

    resources = get_resources()
    graph = build_agent_graph(
        retriever=resources.retriever,
        chat_model=resources.chat_model,
        repository=resources.repository,
        checkpointer=resources.checkpointer,
        permissions=session_permissions(),
    )
    return chat_turn(
        graph,
        question,
        thread_id=st.session_state.thread_id,
        user_id=st.session_state.user_id or None,
        window_size=resources.settings.conversation_window_size,
        preferences=resources.conversations.load_preferences(st.session_state.user_id or None),
    )


def record_turn(turn: Any, question: str) -> None:
    """Append the turn to the displayed list, and to the checkpoint's memory."""

    resources = get_resources()
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": turn.run.answer or "（没有生成回答）",
            "turn": turn,
        }
    )
    st.session_state.pending_action = turn.pending_action
    st.session_state.last_turn = turn
    if turn.paused or turn.status != "answered":
        return
    # 追加到 Checkpoint，使下一轮能读到本轮问答（与 CLI 的 _record_turn 相同做法）
    graph = build_agent_graph(
        retriever=resources.retriever,
        chat_model=resources.chat_model,
        repository=resources.repository,
        checkpointer=resources.checkpointer,
        permissions=session_permissions(),
    )
    graph.update_state(
        {"configurable": {"thread_id": st.session_state.thread_id}},
        {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": turn.run.answer},
            ]
        },
    )
    resources.conversations.touch_thread(st.session_state.thread_id)


def decide(confirmed: bool) -> None:
    """Resume the paused write with an explicit decision, or leave it alone."""

    resources = get_resources()
    graph = build_agent_graph(
        retriever=resources.retriever,
        chat_model=resources.chat_model,
        repository=resources.repository,
        checkpointer=resources.checkpointer,
        permissions=session_permissions(),
    )
    turn = chat_turn(
        graph,
        "",
        thread_id=st.session_state.thread_id,
        user_id=st.session_state.user_id or None,
        resume={"confirmed": confirmed},
    )
    st.session_state.pending_action = None
    st.session_state.last_turn = turn
    label = "已确认并创建工单" if confirmed else "已取消，未写入任何数据"
    st.session_state.flash = f"{label}：{turn.run.answer or turn.status}"


def handle_resume() -> None:
    """Render the confirm/cancel pair for a pending write."""

    pending = st.session_state.pending_action
    if not pending:
        return
    with st.container(border=True):
        st.markdown("**有一个写操作正在等待你的确认**")
        st.markdown(str(pending.get("summary") or pending.get("action")))
        with st.container(horizontal=True, horizontal_alignment="right"):
            if st.button("取消", key="chat_cancel", icon=":material/close:"):
                decide(False)
                st.rerun()
            if st.button(
                "确认创建",
                key="chat_confirm",
                type="primary",
                icon=":material/check:",
            ):
                decide(True)
                st.rerun()


# ---------------------------------------------------------------- page body

resources = get_resources()

if st.session_state.flash:
    st.success(st.session_state.pop("flash"), icon=":material/task_alt:")

if st.session_state.delegated_from:
    st.info(
        f"上一轮被判定为 `{st.session_state.delegated_from}`，而「流式生成」只走知识问答，"
        "查不到设备也建不了工单，因此已自动改用完整智能体回答。"
        "想全程走设备与工单流程时，请关闭上方开关。",
        icon=":material/swap_horiz:",
    )
    st.session_state.delegated_from = None

st.caption(
    "下面显示的是**本次浏览器会话**的对话记录；智能体的记忆保存在 SQLite Checkpoint 里，"
    "按会话编号隔离，刷新页面后仍可继续。"
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        turn = message.get("turn")
        if turn is not None and message["role"] == "assistant":
            render_turn(turn, show_history=False)
        else:
            st.markdown(message["content"])

handle_resume()

stream_mode = st.toggle(
    "流式生成（仅知识问答）",
    value=False,
    help="关闭时走完整智能体：设备查询与工单流程可用；开启时只走知识问答，但能看到逐段生成。",
)

if prompt := st.chat_input("问一个售后问题，例如：D2002 还在保修吗"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if stream_mode:
            turn, _raw = stream_answer(prompt)
            if turn is not None and turn.run.intent_source == "delegated":
                # 流式路径只能答知识问题：这类请求改由完整智能体处理。原因记进会话状态，
                # 由下一趟渲染显示——它描述的是「换了处理路径」，属于页面的说明而不是
                # 这一轮回答的一部分，而且这样刷新后仍然看得见。
                st.session_state.delegated_from = turn.run.intent
                with st.spinner("正在改用完整智能体…"):
                    turn = run_agent_turn(prompt)
        else:
            with st.spinner("正在处理…"):
                turn = run_agent_turn(prompt)
        if turn is None:
            st.error("没有得到结果，请重试。")
        else:
            render_turn(turn, show_history=True)
            record_turn(turn, prompt)
    st.rerun()

with st.expander("这个页面与对话状态的关系", expanded=False):
    st.markdown(
        "- 消息列表存在 `st.session_state`，刷新浏览器就没了：它只是**显示**。\n"
        "- 对话状态存在 `data/rag_agent.sqlite3` 的 Checkpoint 表，按会话编号隔离，"
        "**服务重启后仍能继续**，CLI 的 `rag-agent chat` 用的是同一份数据。\n"
        "- Prompt 只带最近若干条消息，完整历史始终留在 Checkpoint 里。"
    )
    st.caption(f"会话窗口大小：{resources.settings.conversation_window_size} 条")
