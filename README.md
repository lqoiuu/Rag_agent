# 企业售后知识库 RAG 智能体

这是一个面向扫地机器人售后场景的学习与作品集项目。知识资料可以来自有权使用的说明书、维修手册和 FAQ；用户、设备、订单及工单接口均为本地模拟实现，不代表真实企业系统。

当前状态：阶段 3（文档加载与规范化）已完成并通过验收。

## 当前技术基线

- Python 3.13
- uv 与 pyproject.toml 管理项目和依赖
- LangChain、LangGraph、Chroma
- 通义千问，通过 DashScope 的 OpenAI 兼容端点经 httpx 调用
- pypdf 解析 PDF 文本
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

## 尚未实现

- 文档清洗与分片，以及向量入库和检索。
- LangGraph Agent 工作流。
- Streamlit 页面。

模型联网调用需要本地 `.env` 中的 `RAG_AGENT_QWEN_API_KEY`；相关集成测试默认跳过，只有设置 `RAG_AGENT_RUN_LIVE_TESTS=1` 时才访问网络。

具体操作步骤由 Codex 在对话中逐步给出，不在 README 中维护阶段执行流程。
