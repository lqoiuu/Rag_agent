# 企业售后知识库 RAG 智能体

这是一个面向扫地机器人售后场景的学习与作品集项目。知识资料可以来自有权使用的说明书、维修手册和 FAQ；用户、设备、订单及工单接口均为本地模拟实现，不代表真实企业系统。

当前状态：阶段 8（RAG 评测基线）已完成并通过验收。

## 当前技术基线

- Python 3.13
- uv 与 pyproject.toml 管理项目和依赖
- LangChain、LangGraph、Chroma
- 通义千问，通过 DashScope 的 OpenAI 兼容端点经 httpx 调用
- pypdf 解析 PDF 文本，fonttools 补齐 CFF 字体编码解析
- langchain-text-splitters 负责递归字符切分
- pydantic-settings
- pytest、Ruff、mypy

精确依赖版本保存在 uv.lock。真实密钥只允许放入本地 .env，不能提交到 Git。

## 环境重建

- Python 版本：3.13.15，项目解释器位于 `.venv/`，该目录不提交到 Git。
- `pyproject.toml` 要求 `>=3.13,<3.14`，依赖版本以 `uv.lock` 为准，安装时不会静默升级。
- 在 `rag-agent-assistant/` 目录执行：

```powershell
uv sync
.\.venv\Scripts\python.exe -m rag_agent health
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy
```

`uv sync` 会在 `.venv/` 缺失时创建它，并按照 `uv.lock` 安装运行依赖和 dev 依赖。

`rag-agent health` 检查 Python 版本、虚拟环境、依赖版本和核心模块导入；输出 JSON 中 `status` 为 `ok` 时进程退出码为 0，否则为 1。

真实密钥只写入本地 `.env`（从 `.env.example` 复制，已被 `.gitignore` 忽略）。

## 目录职责

- src/rag_agent/：应用源码。
- tests/：自动化测试。
- docs/：稳定的需求、架构、术语和架构决策。
- data/：本地运行数据。`raw/` 存放原始资料，`chroma/` 与 `rag_agent.sqlite3` 保存向量索引与元数据，全部不提交到 Git。
- pyproject.toml：项目元数据、直接依赖和工具配置。
- uv.lock：完整且精确的依赖版本。
- .env.example：环境变量示例，不包含真实密钥。

## 当前可用能力

- 类型化配置和绝对路径解析。
- JSON 结构化日志。
- Python、虚拟环境、依赖版本及核心模块导入健康检查。
- 供应商无关的聊天与 Embedding Provider 协议，以及通义千问适配器。
- 模型调用的超时、重试、错误分类、耗时和 token 用量记录。
- 无网络、无密钥即可运行的 Fake Model 与 Fake Embedding。
- 由配置组装的模型实例，以及最小 Prompt -> Model -> Parser 问答链路。
- `rag-agent ask "问题"` 命令行问答，输出结构化 JSON，模型错误返回退出码 2。
- TXT、Markdown 和 PDF 文档加载，PDF 保留真实页码。
- 编码识别与文本规范化，规范化结果幂等。
- 稳定文档 ID、内容校验值与版本，可追溯每份资料的来源。
- 批量加载的失败隔离、重复内容标记与目录枚举。
- 标题感知的递归字符分片，保留标题层级路径、页码与跨页字符范围，分片可回溯到原文。
- `rag-agent chunk-report <路径>` 参数实验，输出分片数量与长度分布。
- `rag-agent ingest <路径>` 命令行入库，逐文件隔离失败，未变更的文档不会重复调用 Embedding。
- Chroma 持久化向量索引，按稳定 chunk ID 幂等写入，内容变更后自动替换旧分片。
- SQLite 记录文档状态、内容版本历史与每次入库任务。
- `rag-agent reindex` 重建索引并清除已删除文件的残留向量。
- `rag-agent delete-document <来源>` 同时删除向量与元数据。
- Top-K 语义检索，命中项带来源、页码、标题、字符范围、相似度与排名，可按来源过滤。
- `rag-agent search "问题"` 检索调试命令，输出置信判定与判定依据；低置信度返回退出码 1。
- 检索参数（top_k、threshold）可配置，也可在命令行按次覆盖。
- `rag-agent answer "问题"` 带引用回答：只依据编号资料作答，引用编号会校验，没有可核验引用时拒答；支持 `--stream` 真实 Token 流式输出。
- 入库前过滤目录或索引类分片（可用 `--keep-index-chunks` 关闭），减少清单页抢占排名的干扰。
- `rag-agent evaluate --mode retrieval|answer` 离线评测：49 条带预期页码的评测集，输出 Recall@K、MRR、回答率、引用正确率、拒答正确率与忠实度代理指标，并把报告写入 `data/eval/reports/`。

## 当前评测基线

| 指标 | 数值 | 说明 |
|---|---|---|
| Recall@5 | 0.9535 | 43 条可回答问题命中 41 条（按页粒度） |
| MRR | 0.8205 | 正确分片平均排在较前位置 |
| 回答率 | 0.7907 | 提示词消歧后，较修改前 0.6744 提升 |
| 引用正确率 | 0.7941 | 已作答问题中引用页码全部落在预期页的比例 |
| 拒答正确率 | 1.0 | 6 条不可回答问题全部正确拒答 |
| 忠实度（代理） | 0.5556 | 答案为引用分片全文覆盖的比例，属弱指标 |

数据来自 `data/eval/reports/` 下的报告文件，可用同一条命令复现；检索指标确定可重复，回答指标存在运行间波动。

## 尚未实现

- 用户、设备、订单和工单工具。
- LangGraph Agent 工作流。
- Streamlit 页面。
- 矢量轮廓（文字转曲线）PDF 的文本提取，需要 OCR，当前明确不支持。

**已知检索限制**：49 条实测数据显示，可回答与不可回答问题的最高相似度区间重叠 0.1082，前两名分差同样重叠，因此 `RAG_AGENT_RETRIEVAL_THRESHOLD` **不存在能把两组分开的阈值**，它只用于避免无意义的模型调用；拒答由「只能依据资料回答」的约束与引用校验承担。目录类分片已用确定性规则在入库时过滤（前两名分差由 0.0011 提升到 0.0375）。另有两条安全类问题因第 2 页分片过长、主题被稀释而漏检，属已知待改进项。

模型联网调用需要本地 `.env` 中的 `RAG_AGENT_QWEN_API_KEY`；相关集成测试默认跳过，只有设置 `RAG_AGENT_RUN_LIVE_TESTS=1` 时才访问网络。

具体操作步骤由 Codex 在对话中逐步给出，不在 README 中维护阶段执行流程。
