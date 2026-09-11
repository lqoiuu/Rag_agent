"""Entry point for the Streamlit UI.

Run with::

    uv run streamlit run src/rag_agent/ui/app.py

This file is a router and a frame, not a page. Streamlit re-executes it on every rerun
and every page navigation, so everything shared — the resource handles, the demo
identity, the conversation id — is initialised here, before ``page.run()``. Pages then
read it from ``st.session_state`` and stay plain scripts.

Two things this UI deliberately does *not* own:

* **business rules.** Routing, the confirmation gate and citation validation live in
  ``rag_agent.agent``, so the interface cannot bypass the write gate.
* **the authoritative conversation.** ``st.session_state`` holds what is *displayed*;
  the agent's state lives in the SQLite checkpoint and outlives a browser refresh. The
  sidebar shows both counts side by side, because confusing the two is the classic
  Streamlit mistake.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run src/rag_agent/ui/app.py` from a checkout without installing the
# package first. Installed environments already resolve this, so it is a no-op there.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag_agent.tools import ToolPermissions, assign_role  # noqa: E402
from rag_agent.ui.services import default_user_id, get_resources, new_thread_id  # noqa: E402

PAGES = [
    st.Page("app_pages/chat.py", title="对话", icon=":material/chat:", default=True),
    st.Page("app_pages/knowledge.py", title="知识库", icon=":material/library_books:"),
    st.Page("app_pages/search.py", title="检索调试", icon=":material/manage_search:"),
]


def initialise_session() -> None:
    """Initialise every per-session value in one place.

    The sidebar's thread box uses its own widget key, so ``thread_id`` stays a plain
    session value that code may reassign. Sharing one key between a widget and the
    value it edits is what raises ``StreamlitWidgetAlreadyInstantiatedError`` as soon as
    a button tries to change it.
    """

    st.session_state.setdefault("user_id", default_user_id())
    st.session_state.setdefault("thread_id", new_thread_id())
    st.session_state.setdefault("thread_input", st.session_state.thread_id)
    st.session_state.setdefault("role_choice", "end_user")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_action", None)
    st.session_state.setdefault("last_turn", None)
    st.session_state.setdefault("delegated_from", None)
    st.session_state.setdefault("flash", None)


def current_permissions() -> ToolPermissions:
    """The caller's declared identity and role for this session.

    The role is chosen in the sidebar and can only be lowered: ``assign_role`` never raises a
    role above the granted default, so the selector cannot be used to promote oneself even if
    its value were manipulated.
    """

    return ToolPermissions(
        user_id=str(st.session_state.get("user_id") or "") or None,
        role=assign_role(str(st.session_state.get("role_choice") or "")),
    )


def adopt_thread_from_widget() -> None:
    """Copy the sidebar box into the conversation id, on change only."""

    candidate = str(st.session_state.thread_input or "").strip()
    st.session_state.thread_id = candidate or st.session_state.thread_id


def reset_conversation() -> None:
    """Start a new conversation: new thread id, empty displayed history."""

    st.session_state.thread_id = new_thread_id()
    st.session_state.thread_input = st.session_state.thread_id
    st.session_state.messages = []
    st.session_state.pending_action = None
    st.session_state.last_turn = None
    st.session_state.delegated_from = None


def sidebar() -> None:
    """Demo identity, conversation selection and the counters that explain both."""

    resources = get_resources()
    with st.sidebar:
        st.subheader("会话设置")
        st.caption(
            "用户编号只是演示标识，**不是身份认证**：任何人都可以填任意编号。"
            "真实的登录与授权尚未实现。"
        )
        st.text_input("用户编号", key="user_id")
        st.segmented_control(
            "调用角色",
            options=["end_user", "support_agent"],
            key="role_choice",
            help=(
                "由调用方声明，不是模型决定的。end_user 只能读自己的记录；"
                "support_agent 可跨用户读取。角色只能下调，不能上调。"
            ),
        )
        st.text_input(
            "会话编号",
            key="thread_input",
            on_change=adopt_thread_from_widget,
            help="不同编号之间上下文完全隔离；换成别人的编号会被拒绝。",
        )
        st.button(
            "新建会话",
            icon=":material/add_comment:",
            on_click=reset_conversation,
        )

        st.divider()
        st.subheader("当前状态")
        fingerprint = resources.fingerprint()
        st.metric("索引分片", fingerprint["vector_count"])
        st.metric("已入库文档", fingerprint["document_count"])
        st.caption(
            "会话状态由 SQLite Checkpoint 持久化"
            f"（{Path(str(fingerprint['checkpoint_path'])).name}）。"
            "刷新页面不会丢会话，但界面里显示的消息列表会清空，需要重新提问才会再次出现。"
        )
        with st.expander("运行参数"):
            st.json(fingerprint)


def main() -> None:
    """Configure the page, initialise state and run the selected page."""

    st.set_page_config(page_title="企业售后知识库 RAG 智能体", page_icon=":material/support_agent:")
    initialise_session()
    sidebar()
    page = st.navigation(PAGES, position="top")
    page.run()


# Streamlit executes this file on every rerun, so `main()` is called unconditionally
# rather than behind a `__name__ == "__main__"` guard, which would never be true here.
main()
