# 项目结构与文件职责

最后同步：2026-09-10

## 项目边界

RAG_AGENT_PROJECT_PLAN.md 位于外层 Rag_agent 目录，负责保存整个项目的阶段路线和协作规则。

rag-agent-assistant 是实际开发目录，也是 Git 仓库根目录。源码、测试、依赖和稳定设计文档都位于此目录。

外层 Rag_agent 目录还存在一个旧的 .venv，但当前项目不使用它。当前项目只使用 rag-agent-assistant/.venv。

## 当前目录结构

~~~text
rag-agent-assistant/
├── .env.example
├── .gitignore
├── PROJECT_STRUCTURE.md
├── README.md
├── pyproject.toml
├── uv.lock
├── docs/
│   ├── requirements.md
│   ├── architecture.md
│   ├── glossary.md
│   └── adr/
│       └── 0001-technology-stack.md
├── src/
│   └── rag_agent/
│       ├── __init__.py
│       ├── __main__.py
│       ├── health.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       └── observability/
│           ├── __init__.py
│           └── logging.py
└── tests/
    └── unit/
        ├── test_health.py
        └── test_settings.py
~~~

.git、.venv、缓存、字节码和本地运行数据不在结构图中显示。

## 根目录文件

| 文件 | 职责 | 当前作用 |
|---|---|---|
| .env.example | 声明允许使用的环境变量名称 | 提供配置模板，不包含真实密钥 |
| .gitignore | 定义 Git 排除规则 | 防止提交密钥、虚拟环境、缓存、日志和本地数据库 |
| PROJECT_STRUCTURE.md | 保存真实项目结构和职责说明 | 用户要求同步结构时由 Codex 更新 |
| README.md | 说明项目定位、可复现的环境重建步骤、当前能力和边界 | 不保存阶段操作流程 |
| pyproject.toml | 定义项目、Python 范围、直接依赖和工具配置 | uv、pytest、Ruff 和 mypy 的共同配置入口 |
| uv.lock | 锁定完整依赖图 | 保证不同环境安装相同版本 |

## 源码

| 文件 | 职责 | 涉及知识 |
|---|---|---|
| src/rag_agent/__init__.py | 声明 rag_agent Python 包并提供版本号 | 包、模块、包元数据 |
| src/rag_agent/__main__.py | 提供 rag-agent 命令入口 | 命令行参数、进程退出码 |
| src/rag_agent/health.py | 集中检查 Python、虚拟环境、依赖和核心模块导入 | 运行环境、依赖元数据、模块导入 |
| src/rag_agent/config/__init__.py | 对外暴露配置接口 | 包的公共 API |
| src/rag_agent/config/settings.py | 从环境变量读取配置并解析项目绝对路径 | Pydantic、环境配置、路径稳定性 |
| src/rag_agent/observability/__init__.py | 对外暴露日志配置接口 | 模块边界 |
| src/rag_agent/observability/logging.py | 输出 JSON 结构化日志 | Python logging、结构化数据、异常记录 |

## 测试

| 文件 | 验证内容 |
|---|---|
| tests/unit/test_health.py | Python 3.13、项目虚拟环境、依赖安装和核心模块导入 |
| tests/unit/test_settings.py | 相对路径解析、绝对路径稳定性和阶段 1 密钥可选性 |

## 稳定设计文档

| 文件 | 内容 |
|---|---|
| docs/requirements.md | 产品目标、三条业务链路、边界和验收样例 |
| docs/architecture.md | 系统组件、数据流、存储边界和失败降级 |
| docs/glossary.md | RAG、Embedding、Agent、Checkpoint 等术语 |
| docs/adr/0001-technology-stack.md | 技术选型、备选方案和决策后果 |

## 当前阶段

阶段 1（Python 3.13 工程骨架与环境）已完成并通过验收；阶段 2（模型适配层与最小问答）尚未开始。

阶段 1 验收证据（2026-09-10 实际执行）：

- Python 3.13.15，解释器为项目内 `.venv`，由 uv 创建。
- `uv sync --locked`：Resolved 111 packages / Checked 109 packages，锁文件与 pyproject.toml 一致且未被修改。
- `rag-agent health`：`status` 为 `ok`，五项检查全部为 true，五个核心模块导入全部为 ok。
- 锁定依赖版本：chromadb 1.5.9、langchain 1.4.0、langchain-chroma 1.1.0、langgraph 1.2.11、pydantic-settings 2.15.0。
- `pytest`：3 passed。
- `ruff check`：All checks passed；`ruff format --check`：15 files already formatted。
- `mypy`（strict，files = ["src"]）：Success: no issues found in 7 source files。
- Git：`522d030` 建立工程骨架，`cc0b9b9` 补充 README 环境重建说明，工作区干净。

已知环境注意事项：

- 在 DSH 沙箱内运行 pytest 时，pytest 会通过 `tempfile.mkdtemp()` 创建缓存目录（`.venv/Lib/site-packages/_pytest/cacheprovider.py:66`），该目录随后无法被沙箱进程访问或删除，并遗留 `pytest-cache-files-*` 目录。这是沙箱副作用；在普通终端运行 pytest 不受影响。

## 当前已实现能力

- Python 3.13 src 工程布局。
- uv 依赖锁定。
- 类型化环境配置。
- 项目根目录相对路径转绝对路径。
- JSON 结构化日志。
- 环境与核心依赖健康检查。
- 最小单元测试。
- 可复现的环境重建说明（Python 版本、uv sync、健康检查和质量工具命令）。

## 尚未实现

- 通义千问聊天模型与 Embedding Provider。
- 文档解析、清洗和分片。
- Chroma 向量写入、持久化和增量索引。
- 语义检索、低置信度拒答和来源引用。
- 用户、设备、订单和工单工具。
- LangGraph 路由、Checkpoint 和人工确认。
- RAG 离线评测、Streamlit 界面、安全降级和 Docker 交付。
