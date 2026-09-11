"""Retrieval debugging page: why did a query hit those chunks?

The point of this page is the *explanation*, not the list. Stage 6 measured that an
absolute similarity threshold cannot separate answerable from unanswerable questions
(the two score ranges overlap), so ``confident`` and ``margin`` are shown as evidence
for a judgement, never as a correctness verdict. The page says so where the user will
read it, because a bare "0.59" invites exactly the wrong conclusion.
"""

from __future__ import annotations

import streamlit as st

from rag_agent.domain.retrieval import RetrievalResult
from rag_agent.providers.base import ModelError
from rag_agent.ui.services import get_resources

resources = get_resources()
settings = resources.settings

st.caption("检索不会调用聊天模型，可以直接观察分数与排序。这里的结果就是回答环节实际拿到的资料。")

if resources.vectors.count() == 0:
    st.warning(
        "知识库还没有任何分片，检索必然为空。请先到「知识库」页上传资料并入库。",
        icon=":material/warning:",
    )

with st.form("search_form"):
    query = st.text_input("查询语句", value="", placeholder="例如：制造商地址在哪里")
    left, middle, right = st.columns(3)
    with left:
        top_k = st.number_input("Top-K", min_value=1, max_value=20, value=settings.retrieval_top_k)
    with middle:
        threshold = st.number_input(
            "相似度阈值",
            min_value=-1.0,
            max_value=1.0,
            value=float(settings.retrieval_threshold),
            step=0.05,
            help="只用于挡掉接近 0 的无意义匹配，不是拒答机制。",
        )
    with right:
        source = st.text_input("来源过滤（可留空）", value="")
    submitted = st.form_submit_button("检索", type="primary", icon=":material/search:")

if submitted:
    if not query.strip():
        st.error("请先输入查询语句。", icon=":material/error:")
    else:
        result: RetrievalResult | None = None
        try:
            with st.spinner("正在检索…"):
                result = resources.retriever.search(
                    query,
                    top_k=int(top_k),
                    threshold=float(threshold),
                    source=source.strip() or None,
                )
        except ModelError as exc:
            st.error(f"Embedding 调用失败：{exc}", icon=":material/cloud_off:")

        if result is not None:
            if not result.hits:
                st.info(
                    "没有任何分片超过阈值。这通常说明知识库里确实没有相关内容，或者阈值设得过高。",
                    icon=":material/search_off:",
                )
            else:
                best = result.hits[0]
                left, middle, right = st.columns(3)
                left.metric("最高相似度", f"{best.score:.4f}")
                # margin 是前两名分差；只有一个命中时它没有意义，必须显示为「不适用」
                # 而不是拿 None 去格式化。
                middle.metric(
                    "前两名分差",
                    "不适用（只有一个命中）" if result.margin is None else f"{result.margin:.4f}",
                )
                right.metric("置信判定", "是" if result.is_confident else "否")

                st.caption(f"判定依据：{result.reason}")
                with st.expander("为什么分数不能当作「答得对不对」的判据", expanded=False):
                    st.markdown(
                        "49 条评测数据显示：可回答组的最高相似度区间与不可回答组**重叠 0.1082**，"
                        "前两名分差同样重叠。因此不存在能分开两组的单一阈值。\n\n"
                        "阈值在这里只做**成本控制**——避免对接近 0 的匹配调用模型；"
                        "真正的拒答由「只能依据资料回答」的约束与引用校验承担。"
                    )

                st.dataframe(
                    [
                        {
                            "排名": hit.rank,
                            "相似度": round(hit.score, 4),
                            "页码": hit.chunk.page,
                            "来源": hit.chunk.source,
                            "分片": hit.chunk.chunk_id,
                            "字符范围": f"{hit.chunk.char_range[0]}-{hit.chunk.char_range[1]}",
                        }
                        for hit in result.hits
                    ],
                    hide_index=True,
                )

                for hit in result.hits:
                    title = (
                        f"[{hit.rank}] 第 {hit.chunk.page} 页 · "
                        f"{hit.score:.4f} · {hit.chunk.source}"
                    )
                    with st.expander(title):
                        st.markdown(hit.chunk.content)

st.divider()
st.caption(
    f"当前索引共 {resources.vectors.count()} 个分片，阈值默认 {settings.retrieval_threshold}，"
    f"Top-K 默认 {settings.retrieval_top_k}。"
)
