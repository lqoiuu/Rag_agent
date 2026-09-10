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
│       ├── 0001-technology-stack.md
│       └── 0002-provider-layer.md
├── src/
│   └── rag_agent/
│       ├── __init__.py
│       ├── __main__.py
│       ├── health.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── providers.py
│       │   └── settings.py
│       ├── generation/
│       │   ├── __init__.py
│       │   └── minimal_qa.py
│       ├── observability/
│       │   ├── __init__.py
│       │   └── logging.py
│       └── providers/
│           ├── __init__.py
│           ├── base.py
│           ├── fake.py
│           └── qwen.py
└── tests/
    ├── integration/
    │   └── test_qwen_live.py
    └── unit/
        ├── model_test_support.py
        ├── test_fake_providers.py
        ├── test_health.py
        ├── test_minimal_chain.py
        ├── test_provider_errors.py
        ├── test_provider_factory.py
        ├── test_qwen_adapter.py
        └── test_settings.py
~~~

.git、.venv、缓存、字节码和本地运行数据不在结构图中显示。

## 根目录文件

| 文件 | 职责 | 当前作用 |
|---|---|---|
| .env.example | 声明允许使用的环境变量名称 | 提供配置模板，含路径、模型和重试参数，不包含真实密钥 |
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
| src/rag_agent/config/__init__.py | 对外暴露配置与 Provider 组装接口 | 包的公共 API |
| src/rag_agent/config/settings.py | 从环境变量读取配置并解析项目绝对路径，含模型名、超时和重试参数 | Pydantic、环境配置、路径稳定性 |
| src/rag_agent/config/providers.py | 依据 Settings 组装通义千问聊天与 Embedding 实例，缺密钥时在发请求前失败 | 依赖注入、组装与配置边界 |
| src/rag_agent/providers/__init__.py | 对外暴露协议、错误类型、真实实现和 Fake 实现 | 模块边界、公共 API |
| src/rag_agent/providers/base.py | 定义 ChatMessage、ChatResponse、EmbeddingResponse 等值对象和 ChatModel、EmbeddingModel 协议，以及七类模型错误 | 依赖倒置、结构化类型、错误分类 |
| src/rag_agent/providers/qwen.py | 通过 httpx 调用 DashScope OpenAI 兼容端点，实现超时、指数退避重试、状态码分类、耗时与 token 记录 | HTTP 客户端、重试策略、密钥外置 |
| src/rag_agent/providers/fake.py | 提供脚本化 FakeChatModel 和确定性 FakeEmbeddingModel | 测试替身、确定性测试 |
| src/rag_agent/generation/__init__.py | 对外暴露最小问答链路接口 | 模块边界 |
| src/rag_agent/generation/minimal_qa.py | 用 LCEL 组装 Prompt -> Model -> Parser，并返回带模型证据的结构化答案 | LCEL、Runnable 协议、结构化输出 |
| src/rag_agent/observability/__init__.py | 对外暴露日志配置接口 | 模块边界 |
| src/rag_agent/observability/logging.py | 输出 JSON 结构化日志 | Python logging、结构化数据、异常记录 |

## 测试

| 文件 | 验证内容 |
|---|---|
| tests/unit/model_test_support.py | 共享的无网络测试辅助：脚本化 HTTP 传输、模型构造器和响应构造器 |
| tests/unit/test_health.py | Python 3.13、项目虚拟环境、依赖安装和核心模块导入 |
| tests/unit/test_settings.py | 相对路径解析、模型配置默认值与环境变量覆盖、非法值被拒和密钥可选性 |
| tests/unit/test_fake_providers.py | Fake 模型的响应顺序、消息记录、异常注入、Embedding 确定性与边界校验 |
| tests/unit/test_provider_errors.py | 状态码到错误类型的映射、重试次数、不可重试错误、错误信息不泄露密钥 |
| tests/unit/test_qwen_adapter.py | 请求 URL、鉴权头、请求体、响应解析、批次数与维度校验、客户端所有权 |
| tests/unit/test_provider_factory.py | 配置到模型实例的组装：模型名、base_url、重试上限和缺密钥行为 |
| tests/unit/test_minimal_chain.py | 提示词渲染、结构化答案证据、链路复用和协议可替换性 |
| tests/integration/test_qwen_live.py | 真实模型联网调用与 Embedding 维度一致性，默认跳过 |

## 稳定设计文档

| 文件 | 内容 |
|---|---|
| docs/requirements.md | 产品目标、三条业务链路、边界和验收样例 |
| docs/architecture.md | 系统组件、数据流、存储边界和失败降级 |
| docs/glossary.md | RAG、Embedding、Agent、Checkpoint 等术语 |
| docs/adr/0001-technology-stack.md | 技术选型、备选方案和决策后果 |
| docs/adr/0002-provider-layer.md | 模型接入方式、同步优先取舍和错误分类决策 |

## 当前阶段

阶段 1（Python 3.13 工程骨架与环境）已完成并通过验收。

阶段 2（模型适配层与最小问答）已完成并通过验收，含真实模型联网验证。

阶段 1 证据（2026-09-10 实际执行）：

- Python 3.13.15，解释器为项目内 `.venv`，由 uv 创建。
- `uv sync --locked`：Resolved 111 packages / Checked 109 packages，锁文件与 pyproject.toml 一致。
- `rag-agent health`：`status` 为 `ok`，五项检查全部为 true。
- 完成时的测试结果为 3 passed。
- Git：`522d030` 建立工程骨架，`cc0b9b9` 补充 README 环境重建说明。

阶段 2 证据（2026-09-10 实际执行）：

- 代码与配置提交：`fc14919`。
- `pytest`：58 passed, 2 skipped（跳过的是需要联网和密钥的集成测试）。
- `ruff check`：All checks passed；`ruff format --check`：23 files already formatted。
- `mypy`（strict，files = ["src"]）：Success: no issues found in 14 source files。
- `rag-agent health`：`status` 仍为 `ok`，新增导入链未破坏命令行入口。
- `uv.lock` 已同步 httpx 直接依赖（`>=0.28,<1`）。
- 锁定依赖版本：chromadb 1.5.9、langchain 1.4.0、langchain-chroma 1.1.0、langgraph 1.2.11、pydantic-settings 2.15.0。
- 真实模型联网验证（2026-09-10 实际执行）：`RAG_AGENT_RUN_LIVE_TESTS=1` 下 `pytest tests/integration` 为 2 passed in 3.90s；最小链路真实调用返回 `model=qwen-plus`、`latency_ms=1323`、`attempts=1`。
- 验收对照：真实模型可以回答固定问题；无网络环境依靠 Fake Model 稳定运行全部核心测试（58 passed, 2 skipped）。

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
- 供应商无关的聊天与 Embedding Provider 协议，以及通义千问适配器。
- 模型调用的超时、重试、错误分类、耗时和 token 用量记录。
- 无网络、无密钥即可运行的 Fake Chat 与 Fake Embedding。
- 由 Settings 组装的模型实例，以及最小 Prompt -> Model -> Parser 问答链路。

## 尚未实现

- 文档解析、清洗和分片。
- Chroma 向量写入、持久化和增量索引。
- 语义检索、低置信度拒答和来源引用。
- 用户、设备、订单和工单工具。
- LangGraph 路由、Checkpoint 和人工确认。
- RAG 离线评测、Streamlit 界面、安全降级和 Docker 交付。

## 最近结构变化

本次同步（阶段 2）相对上一次的主要变化：

- 新增 `src/rag_agent/providers/`，承载协议、通义千问适配器和 Fake 实现。
- 新增 `src/rag_agent/generation/`，承载最小 LCEL 问答链路。
- 新增 `src/rag_agent/config/providers.py`，把 Settings 到模型实例的组装集中到配置层。
- 新增 `tests/integration/`，用于默认跳过的真实联网验证。
- `tests/unit/` 新增四个模型层测试文件和一个共享测试辅助模块。
- `pyproject.toml` 增加 httpx 直接依赖和中文标点白名单。
- 新增 `docs/adr/0002-provider-layer.md`，记录模型接入层的实现方式与取舍。
