"""Headless tests for the Streamlit UI, using Streamlit's own AppTest.

``AppTest`` executes the app in-process, so these catch import errors, bad widget
arguments and exceptions that a running-server check would never surface. Two things
make them hermetic:

* every store is pointed at a temporary directory through the same environment
  variables the CLI uses, so nothing touches the developer's real index or database;
* the resource layer is replaced with one built on the scripted fake models, so a test
  can never reach the network. An earlier version of this file used a placeholder API
  key and really did call the provider — the 401 is exactly why the injection exists.

What this cannot cover is stated rather than faked: chart and dataframe *selections*,
custom component JavaScript and anything that only exists in a rendered browser are
outside ``AppTest``; the visual result still needs a human looking at the page.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from agent_test_support import MANUAL_PAGE_27, make_retriever
from streamlit.testing.v1 import AppTest

from rag_agent.config.settings import Settings
from rag_agent.memory import ConversationStore, SQLiteCheckpointer
from rag_agent.providers.fake import FakeChatModel, FakeEmbeddingModel
from rag_agent.storage import BusinessRepository, MetadataStore
from rag_agent.ui import services
from rag_agent.vectorstore import ChunkVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP = PROJECT_ROOT / "src" / "rag_agent" / "ui" / "app.py"


def build_test_resources(tmp_path: Path, *replies: str) -> services.AppResources:
    """Real stores and graph wiring, fake models: no network, no real data."""

    data_dir = tmp_path / "data"
    (data_dir / "business").mkdir(parents=True, exist_ok=True)
    (data_dir / "business" / "seed.json").write_text(
        (PROJECT_ROOT / "data" / "business" / "seed.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    settings = Settings(
        data_dir=data_dir,
        chroma_dir=tmp_path / "chroma",
        sqlite_path=tmp_path / "rag_agent.sqlite3",
        checkpoint_path=tmp_path / "rag_agent.sqlite3",
        qwen_api_key="not-used",
    )
    repository = BusinessRepository(settings.sqlite_path)
    repository.seed_from_file(data_dir / "business" / "seed.json")
    return services.AppResources(
        settings=settings,
        vectors=ChunkVectorStore(settings.chroma_dir),
        metadata=MetadataStore(settings.sqlite_path),
        embedding_model=FakeEmbeddingModel(dimension=16),
        chat_model=FakeChatModel(list(replies)),
        checkpointer=SQLiteCheckpointer(settings.checkpoint_path),
        conversations=ConversationStore(settings.checkpoint_path),
        repository=repository,
        retriever=make_retriever(MANUAL_PAGE_27),
    )


@pytest.fixture
def ui(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Any]:
    """Install a factory that swaps the fake-model resources into the running app."""

    installed: list[services.AppResources] = []

    def install(*replies: str) -> services.AppResources:
        resources = build_test_resources(tmp_path, *replies)
        installed.append(resources)
        monkeypatch.setattr(services, "get_resources", lambda: resources)
        monkeypatch.setattr("rag_agent.ui.app.get_resources", lambda: resources)
        return resources

    yield install

    for resources in installed:
        resources.close()
    st.cache_resource.clear()


def run_app(page: str | None = None) -> AppTest:
    """Start the app at the entry point, optionally switching to one page."""

    app = AppTest.from_file(str(APP), default_timeout=120).run()
    if page is not None:
        app = app.switch_page(page).run()
    return app


def exceptions(app: AppTest) -> list[str]:
    return [str(item.value) for item in app.exception]


def test_entry_point_renders_the_chat_page(ui: Any) -> None:
    ui()

    app = run_app()

    assert not app.exception, exceptions(app)
    assert app.session_state["user_id"] == services.default_user_id()
    assert app.session_state["thread_id"].startswith("T")


def test_every_page_renders_without_an_empty_index(ui: Any) -> None:
    """An empty knowledge base is the first thing a new user sees; it must not crash."""

    ui()
    for page in ("app_pages/chat.py", "app_pages/knowledge.py", "app_pages/search.py"):
        app = run_app(page)
        assert not app.exception, f"{page}: {exceptions(app)}"


def test_search_page_warns_when_the_index_is_empty(ui: Any) -> None:
    ui()

    app = run_app("app_pages/search.py")

    assert any("知识库还没有任何分片" in warning.value for warning in app.warning)


def test_knowledge_page_reports_an_empty_library(ui: Any) -> None:
    ui()

    app = run_app("app_pages/knowledge.py")

    assert any("已入库文档（0）" in block.value for block in app.subheader)
    assert any("知识库还是空的" in info.value for info in app.info)


def test_new_conversation_button_clears_the_displayed_messages(ui: Any) -> None:
    ui()
    app = run_app()
    app.session_state["messages"] = [{"role": "user", "content": "旧消息"}]
    app.run()
    original = app.session_state["thread_id"]

    app.sidebar.button[0].click().run()

    assert app.session_state["thread_id"] != original
    assert app.session_state["messages"] == []


def test_device_question_reaches_the_business_tools(ui: Any) -> None:
    """A device question must call the tools, not answer from knowledge alone."""

    from agent_test_support import intent_reply

    ui(intent_reply("device"))

    app = run_app()
    app.chat_input[0].set_value("D2002 还在保修吗").run()

    assert not app.exception, exceptions(app)
    turn = app.session_state["messages"][1]["turn"]
    assert turn.run.tool_results[0]["tool"] == "device.lookup"
    assert "2028-01-10" in turn.run.answer


def test_ticket_request_pauses_and_cancelling_writes_nothing(ui: Any) -> None:
    """The write gate must be visible in the UI, and cancel must leave no row."""

    from agent_test_support import extract_reply, intent_reply

    resources = ui(
        intent_reply("ticket"),
        extract_reply(device_id="D2001", issue="主刷一直卡住", contact="13800000001"),
    )

    app = run_app()
    app.chat_input[0].set_value("帮我把 D2001 报修，主刷一直卡住，联系我 13800000001").run()

    assert not app.exception, exceptions(app)
    assert app.session_state["pending_action"] is not None
    assert app.session_state["pending_action"]["action"] == "ticket.create"
    assert resources.repository.list_tickets() == ()

    app.button(key="chat_cancel").click().run()

    assert not app.exception, exceptions(app)
    assert app.session_state["pending_action"] is None
    assert resources.repository.list_tickets() == ()


def test_confirming_creates_exactly_one_ticket(ui: Any) -> None:
    from agent_test_support import extract_reply, intent_reply

    resources = ui(
        intent_reply("ticket"),
        extract_reply(device_id="D2001", issue="主刷一直卡住", contact="13800000001"),
    )

    app = run_app()
    app.chat_input[0].set_value("帮我把 D2001 报修，主刷一直卡住，联系我 13800000001").run()
    app.button(key="chat_confirm").click().run()

    assert not app.exception, exceptions(app)
    assert len(resources.repository.list_tickets()) == 1


def test_upload_writes_the_file_and_ingests_it(ui: Any) -> None:
    """The upload path must reach the real ingestion pipeline."""

    resources = ui()
    document = (
        "用户手册\n\n## 安全提示\n\n请勿在潮湿地面使用本产品。\n\n"
        "## 充电说明\n\n请使用原装充电座，充电时间约 6.5 小时。\n"
    )

    app = run_app("app_pages/knowledge.py")
    app.file_uploader[0].set_value(("manual.md", document.encode("utf-8"), "text/markdown"))
    app.button[0].click().run()

    assert not app.exception, exceptions(app)
    assert list((resources.settings.data_dir / "raw").glob("manual.md"))
    assert resources.metadata.get_document("manual.md") is not None


def test_search_page_reports_hits_for_an_indexed_chunk(ui: Any) -> None:
    """With content in the index, the page shows the explanation, not just a score."""

    ui()  # 测试用的 retriever 已经装入了 MANUAL_PAGE_27 这一条分片

    app = run_app("app_pages/search.py")
    app.text_input[0].set_value(MANUAL_PAGE_27).run()
    app.button[0].click().run()

    assert not app.exception, exceptions(app)
    metrics = {metric.label: metric.value for metric in app.metric}
    assert "最高相似度" in metrics
    assert "前两名分差" in metrics
    assert any("判定依据" in caption.value for caption in app.caption)


def test_streaming_toggle_uses_the_knowledge_path(ui: Any) -> None:
    """With streaming on, a knowledge turn runs through the streaming path.

    That path reads memory without writing a checkpoint, so it reports
    ``checkpoints_after == checkpoints_before``. The difference is asserted rather than
    papered over.
    """

    from agent_test_support import answer_reply, intent_reply

    ui(intent_reply("knowledge"), answer_reply(MANUAL_PAGE_27))

    app = run_app()
    app.toggle[0].set_value(True).run()
    app.chat_input[0].set_value(MANUAL_PAGE_27).run()

    assert not app.exception, exceptions(app)
    turn = app.session_state["messages"][1]["turn"]
    assert turn.run.intent_source == "stream"
    assert turn.checkpoints_after == turn.checkpoints_before


def test_streaming_hands_a_device_question_to_the_full_agent(ui: Any) -> None:
    """The regression from the running UI: a device question must not be refused.

    Streaming cannot reach the device tools, so it used to answer from the knowledge base
    and refuse with "the documents do not mention D2002" — which reads as a missing
    document rather than as a mode limitation. The page must now say what happened and
    answer through the full agent.
    """

    from agent_test_support import intent_reply

    ui(intent_reply("device"), intent_reply("device"))

    app = run_app()
    app.toggle[0].set_value(True).run()
    app.chat_input[0].set_value("D2002 还在保修吗").run()

    assert not app.exception, exceptions(app)
    notices = [info.value for info in app.info]
    assert any("只走知识问答" in text for text in notices), notices
    assert any("device" in text for text in notices), notices

    final = app.session_state["last_turn"]
    assert final.run.intent_source != "delegated"
    assert "2028-01-10" in final.run.answer
    assert final.turn_tool_results[0]["tool"] == "device.lookup"
