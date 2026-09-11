"""Knowledge base page: upload, inspect, delete and rebuild the index.

Ingestion is idempotent by construction — the pipeline short-circuits an unchanged
document and never re-embeds it — so this page does not need to track what it has
already done. That is why it re-reads the truth from SQLite on every rerun instead of
trusting anything cached in the session: a stale list would be worse than a slow one.

The one thing that *does* need care is not repeating work on every rerun. Uploaded
files are written and ingested inside the submit handler, and the result is carried to
the next render in a one-shot session value that the page pops, so a rerun cannot
ingest twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from rag_agent.ingestion import ingest_path, remove_document, sync_index
from rag_agent.storage import utc_now
from rag_agent.ui.services import get_resources, raw_directory

resources = get_resources()
settings = resources.settings

st.caption(
    "上传的文件写入 `data/raw/` 后立即入库。入库是幂等的：内容没变的文档会被跳过，"
    "不会重复调用 Embedding，也不会产生重复分片。"
)

flash = st.session_state.pop("kb_flash", None)
if flash:
    st.success(flash, icon=":material/check:")

with st.form("kb_upload", clear_on_submit=True):
    uploads = st.file_uploader(
        "上传资料（TXT / Markdown / PDF）",
        type=["txt", "md", "markdown", "pdf"],
        accept_multiple_files=True,
        key="kb_files",
    )
    submitted = st.form_submit_button("上传并入库", type="primary", icon=":material/upload:")

if submitted and uploads:
    target_dir = raw_directory(settings)
    outcomes: list[str] = []
    for upload in uploads:
        destination = target_dir / Path(upload.name).name
        destination.write_bytes(upload.getvalue())
        result = ingest_path(
            destination,
            store=resources.metadata,
            vectors=resources.vectors,
            embedding_model=resources.embedding_model,
            root=target_dir,
        )
        if result.ok:
            outcomes.append(
                f"{destination.name}：{result.status}，分片 {result.chunk_count} 个，"
                f"向量 {result.vector_count} 个"
            )
        else:
            outcomes.append(f"{destination.name}：失败（{result.error_code}）")
    st.session_state.kb_flash = "入库完成 —— " + "；".join(outcomes)

st.divider()

documents = resources.metadata.list_documents()
st.subheader(f"已入库文档（{len(documents)}）")

if not documents:
    st.info(
        "知识库还是空的。上传一份说明书后，检索与对话才有资料可用。",
        icon=":material/info:",
    )
else:
    for document in documents:
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.markdown(f"**{document.source}**")
                st.caption(
                    f"类型 {document.file_type} · {document.page_count} 页 · "
                    f"{document.chunk_count} 分片 · 版本 {document.version} · "
                    f"更新于 {document.updated_at}"
                )
                st.badge(
                    f"{resources.vectors.count_for_document(document.document_id)} 个向量",
                    icon=":material/dataset:",
                    color="blue",
                )
            with right:
                if st.button(
                    "删除",
                    key=f"kb_delete_{document.document_id}",
                    icon=":material/delete:",
                ):
                    removed = remove_document(
                        document.source, store=resources.metadata, vectors=resources.vectors
                    )
                    st.session_state.kb_flash = (
                        f"已删除 {document.source}，同时清除 {removed.vector_count} 个向量"
                    )
                    st.rerun()

st.divider()
with st.container(horizontal=True):
    if st.button("重建索引", icon=":material/refresh:"):
        results = sync_index(
            raw_directory(settings),
            store=resources.metadata,
            vectors=resources.vectors,
            embedding_model=resources.embedding_model,
        )
        indexed = sum(1 for result in results if result.status == "indexed")
        unchanged = sum(1 for result in results if result.status == "unchanged")
        failed = [result for result in results if not result.ok]
        st.session_state.kb_flash = (
            f"重建完成：重新入库 {indexed} 份，跳过未变更 {unchanged} 份，失败 {len(failed)} 份"
        )
        st.rerun()

with st.expander("最近的入库任务", expanded=False):
    jobs = resources.metadata.list_jobs(limit=10)
    if not jobs:
        st.caption("本次尚无任务记录。")
    for job in jobs:
        st.markdown(
            f"- `{job.started_at}` **{job.source}** → {job.status}"
            + (f"（{job.error_code}）" if job.error_code else "")
        )

with st.expander("数据边界", expanded=False):
    st.markdown(
        "知识资料来自你有权使用的说明书等文件；**用户、设备、订单与工单都是模拟数据**，"
        "不代表真实个人或企业信息。上传的资料只写入本地 `data/raw/`，不提交到 Git。"
    )
    st.caption(f"当前时间：{utc_now()}")
    st.json(json.loads(json.dumps(resources.fingerprint(), ensure_ascii=False, default=str)))
