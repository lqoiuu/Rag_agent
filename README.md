# 企业售后知识库 RAG 智能体

这是一个面向扫地机器人售后场景的学习与作品集项目。知识资料可以来自有权使用的说明书、维修手册和 FAQ；用户、设备、订单及工单接口均为本地模拟实现，不代表真实企业系统。

当前状态：阶段 1，Python 3.13 工程骨架与环境。

## 当前技术基线

- Python 3.13
- uv 与 pyproject.toml 管理项目和依赖
- LangChain、LangGraph、Chroma
- pydantic-settings
- pytest、Ruff、mypy

精确依赖版本保存在 uv.lock。真实密钥只允许放入本地 .env，不能提交到 Git。

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

## 尚未实现

- 通义千问模型适配。
- 文档解析、分片、向量入库和检索。
- LangGraph Agent 工作流。
- Streamlit 页面。

具体操作步骤由 Codex 在对话中逐步给出，不在 README 中维护阶段执行流程。
