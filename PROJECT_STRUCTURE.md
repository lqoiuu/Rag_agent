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
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── documents.py
│       │   └── errors.py
│       ├── generation/
│       │   ├── __init__.py
│       │   └── minimal_qa.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loaders.py
│       │   ├── normalize.py
│       │   ├── pipeline.py
│       │   ├── splitters.py
│       │   └── stats.py
│       ├── observability/
│       │   ├── __init__.py
│       │   └── logging.py
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── fake.py
│       │   └── qwen.py
│       ├── storage/
│       │   ├── __init__.py
│       │   └── sqlite.py
│       └── vectorstore/
│           ├── __init__.py
│           └── chroma.py
└── tests/
    ├── integration/
    │   └── test_qwen_live.py
    └── unit/
        ├── ingestion_test_support.py
        ├── model_test_support.py
        ├── pdf_fixtures.py
        ├── test_chroma_store.py
        ├── test_chunk_stats.py
        ├── test_cli_ask.py
        ├── test_cli_chunk_report.py
        ├── test_cli_ingest.py
        ├── test_documents.py
        ├── test_fake_providers.py
        ├── test_health.py
        ├── test_ingest_pipeline.py
        ├── test_loaders.py
        ├── test_loaders_pdf.py
        ├── test_logging.py
        ├── test_minimal_chain.py
        ├── test_normalize.py
        ├── test_provider_errors.py
        ├── test_provider_factory.py
        ├── test_qwen_adapter.py
        ├── test_settings.py
        ├── test_sqlite_store.py
        └── test_splitters.py
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
| src/rag_agent/__main__.py | 提供 `health`、`ask`、`chunk-report`、`ingest`、`reindex`、`delete-document` 六个命令，退出码 0 成功、1 部分失败、2 错误 | 命令行参数、进程退出码、错误到退出码的映射 |
| src/rag_agent/health.py | 集中检查 Python、虚拟环境、依赖和核心模块导入 | 运行环境、依赖元数据、模块导入 |
| src/rag_agent/config/__init__.py | 对外暴露配置与 Provider 组装接口 | 包的公共 API |
| src/rag_agent/config/settings.py | 从环境变量读取配置并解析项目绝对路径，含模型名、超时和重试参数 | Pydantic、环境配置、路径稳定性 |
| src/rag_agent/config/providers.py | 依据 Settings 组装通义千问聊天与 Embedding 实例，缺密钥时在发请求前失败 | 依赖注入、组装与配置边界 |
| src/rag_agent/domain/__init__.py | 对外暴露领域模型与摄取错误 | 包的公共 API |
| src/rag_agent/domain/documents.py | 定义 FileType、DocumentStatus、DocumentPage、SourceDocument、DocumentChunk，以及稳定文档 ID、校验值和版本派生 | 领域模型、稳定标识、内容版本 |
| src/rag_agent/domain/errors.py | 定义八类带稳定 code 的摄取错误，含分片错误 | 错误隔离、异常设计 |
| src/rag_agent/ingestion/__init__.py | 对外暴露加载、规范化、分片、统计与入库接口 | 模块边界 |
| src/rag_agent/ingestion/normalize.py | 识别编码并规范化文本：BOM 嗅探、编码回退、换行与控制字符处理、空行折叠、幂等保证 | 字符编码、文本规范化、幂等性 |
| src/rag_agent/ingestion/loaders.py | 把 TXT、Markdown 和 PDF 加载为 SourceDocument，批量失败隔离、重复内容标记与目录枚举 | Document Loader、错误隔离、批量一致性 |
| src/rag_agent/ingestion/splitters.py | 标题感知与递归字符分片：按标题切 section、保留标题层级路径、计算跨页字符范围、强制分片可回溯到原文 | Text Splitter、分片参数、字符偏移、不变量校验 |
| src/rag_agent/ingestion/stats.py | 汇总分片数量与长度分布，含最近秩分位数和直方图，供参数实验比较 | 分布统计、分位数、实验可重复性 |
| src/rag_agent/ingestion/pipeline.py | 串起加载、分片、分批 Embedding、写向量与写元数据；未变更短路、内容变更替换旧分片、失败记录任务且保留旧版本 | 幂等性、增量更新、两存储一致性 |
| src/rag_agent/generation/__init__.py | 对外暴露最小问答链路接口 | 模块边界 |
| src/rag_agent/generation/minimal_qa.py | 用 LCEL 组装 Prompt -> Model -> Parser，并返回带模型证据的结构化答案 | LCEL、Runnable 协议、结构化输出 |
| src/rag_agent/observability/__init__.py | 对外暴露日志配置接口 | 模块边界 |
| src/rag_agent/observability/logging.py | 输出 JSON 结构化日志，并把 httpx 的 INFO 日志降为 WARNING | Python logging、结构化数据、异常记录 |
| src/rag_agent/providers/__init__.py | 对外暴露协议、错误类型、真实实现和 Fake 实现 | 模块边界、公共 API |
| src/rag_agent/providers/base.py | 定义 ChatMessage、ChatResponse、EmbeddingResponse 等值对象和 ChatModel、EmbeddingModel 协议，以及七类模型错误 | 依赖倒置、结构化类型、错误分类 |
| src/rag_agent/providers/qwen.py | 通过 httpx 调用 DashScope OpenAI 兼容端点，实现超时、指数退避重试、状态码分类、耗时与 token 记录 | HTTP 客户端、重试策略、密钥外置 |
| src/rag_agent/providers/fake.py | 提供脚本化 FakeChatModel 和确定性 FakeEmbeddingModel | 测试替身、确定性测试 |
| src/rag_agent/storage/__init__.py | 对外暴露元数据仓储接口 | 模块边界 |
| src/rag_agent/storage/sqlite.py | SQLite 元数据存储：documents、document_versions、ingestion_jobs 三张表，按来源 upsert、查询、删除并记录任务 | sqlite3、事务、唯一约束、版本历史 |
| src/rag_agent/vectorstore/__init__.py | 对外暴露向量存储接口 | 模块边界 |
| src/rag_agent/vectorstore/chroma.py | Chroma 封装：按稳定 chunk ID 幂等 upsert、按文档取 ID 与内容、删除、向量查询，Embedding 由项目 Provider 提供 | 向量维度、距离度量、幂等写入 |

## 测试

| 文件 | 验证内容 |
|---|---|
| tests/unit/ingestion_test_support.py | 无文件系统的文档、分片、固定分片器与隔离向量存储构造辅助 |
| tests/unit/model_test_support.py | 无网络测试辅助：脚本化 HTTP 传输、模型构造器和响应构造器 |
| tests/unit/pdf_fixtures.py | 自行组装对象与交叉引用表的最小 PDF 构造器，不依赖 pypdf 修复破损文件 |
| tests/unit/test_health.py | Python 3.13、项目虚拟环境、依赖安装和核心模块导入 |
| tests/unit/test_settings.py | 相对路径解析、模型配置默认值与环境变量覆盖、非法值被拒和密钥可选性 |
| tests/unit/test_logging.py | JSON 日志格式、handler 幂等安装、httpx 日志降级和异常信息记录 |
| tests/unit/test_fake_providers.py | Fake 模型的响应顺序、消息记录、异常注入、Embedding 确定性与边界校验 |
| tests/unit/test_provider_errors.py | 状态码到错误类型的映射、重试次数、不可重试错误、错误信息不泄露密钥 |
| tests/unit/test_qwen_adapter.py | 请求 URL、鉴权头、请求体、响应解析、批次数与维度校验、客户端所有权 |
| tests/unit/test_provider_factory.py | 配置到模型实例的组装：模型名、base_url、重试上限和缺密钥行为 |
| tests/unit/test_minimal_chain.py | 提示词渲染、结构化答案证据、链路复用和协议可替换性 |
| tests/unit/test_cli_ask.py | ask 命令的成功输出、用法错误、供应商错误到退出码的映射 |
| tests/unit/test_documents.py | 文件类型映射、稳定文档 ID、内容版本、状态派生、不可变性和分片 ID |
| tests/unit/test_normalize.py | 解码回退、BOM、控制字符、空行折叠、NBSP 和规范化幂等性 |
| tests/unit/test_loaders.py | 文本加载、来源相对路径、六类边界错误、批量失败隔离、重复标记和目录枚举 |
| tests/unit/test_loaders_pdf.py | PDF 多页提取、无文本、加密和损坏文件，需要 pypdf |
| tests/unit/test_splitters.py | 标题层级路径、未命名段落、分片编号、跨页字符范围、重叠定位、空分片过滤、非法配置、缺失依赖、LangChain 分片器的长度上限 |
| tests/unit/test_chunk_stats.py | 分片计数、最小/中位/均值/最大长度、最近秩 p90、直方图分桶和空集合 |
| tests/unit/test_cli_chunk_report.py | chunk-report 的用法错误、摄取错误、非法参数与多组合实验输出 |
| tests/unit/test_sqlite_store.py | 元数据往返、版本历史、同版本幂等、任务记录与排序、删除、文件持久化 |
| tests/unit/test_chroma_store.py | 向量幂等写入、同 ID 内容替换、按文档删除隔离、查询排序、输入长度校验 |
| tests/unit/test_ingest_pipeline.py | 未变更短路不调用 Embedding、内容变更替换旧分片、失败保留旧版本并记录任务、删除清理两个存储、分批 Embedding |
| tests/unit/test_cli_ingest.py | ingest、reindex、delete-document 的参数校验、退出码与成功、部分失败、缺密钥分支 |
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

阶段 3（文档加载与规范化）已完成并通过验收。

阶段 4（文本清洗、分片与参数实验）已完成并通过验收。

阶段 5（向量化、持久化与增量索引）已完成并通过验收，含真实说明书入库验证。

阶段 1 证据（2026-09-10 实际执行）：

- Python 3.13.15，解释器为项目内 `.venv`，由 uv 创建。
- `uv sync --locked`：Resolved 111 packages / Checked 109 packages，锁文件与 pyproject.toml 一致。
- `rag-agent health`：`status` 为 `ok`，五项检查全部为 true。
- 完成时的测试结果为 3 passed。
- Git：`522d030` 建立工程骨架，`cc0b9b9` 补充 README 环境重建说明。

阶段 2 证据（2026-09-10 实际执行）：

- 代码与配置提交：`fc14919`。
- `pytest`：58 passed, 2 skipped（跳过的是需要联网和密钥的集成测试）。
- `mypy`（strict）：Success: no issues found in 14 source files。
- `uv.lock` 已同步 httpx 直接依赖（`>=0.28,<1`）。
- 锁定依赖版本：chromadb 1.5.9、langchain 1.4.0、langchain-chroma 1.1.0、langgraph 1.2.11、pydantic-settings 2.15.0。
- 真实模型联网验证：`RAG_AGENT_RUN_LIVE_TESTS=1` 下 `pytest tests/integration` 为 2 passed in 3.90s；最小链路真实调用返回 `model=qwen-plus`、`latency_ms=1323`、`attempts=1`。
- Git：`29b1590` 同步结构文档与 README，`fa45a09` 记录 ADR-0002 与阶段结论。

阶段 3 证据（2026-09-10 实际执行）：

- 代码与配置提交：`b641ebc`。
- `pytest`（学习者在本地执行）：101 passed, 2 skipped in 2.20s，其中四个 PDF 用例首次真实执行并通过。
- `ruff check`：All checks passed；`ruff format --check`：36 files already formatted。
- `mypy`（strict，files = ["src"]）：Success: no issues found in 20 source files。
- 真实模型验证：`rag-agent ask "扫地机器人充不进电时应该先检查什么。"` 返回 `status=ok`、`model=qwen-plus`、`latency_ms=1244.1`、`attempts=1`、`prompt_tokens=64`、`completion_tokens=33`。
- `uv.lock` 已同步 pypdf 6.18.0。
- 提交后追加了 `tests/unit/test_logging.py`（四个用例）；学习者在本地复核完整套件为 105 passed, 2 skipped in 1.18s。
- 已知限制：编码检测是启发式的，`gb18030` 先于 `big5` 尝试，个别在两种编码下都合法的字节序列会被解成错字而不报错。

阶段 4 证据（2026-09-10 实际执行）：

- `ruff check`：All checks passed；`ruff format --check`：42 files already formatted。
- `mypy`（strict，files = ["src"]）：Success: no issues found in 22 source files。
- `pytest`：122 passed, 2 skipped（在 DSH 沙箱内执行，含四个 LangChain 分片用例）；另有 17 个使用 `tmp_path` 的用例受沙箱限制无法在此运行，它们已由学习者在阶段 3 的本地运行中确认通过，两处合计 139 passed, 2 skipped。
- `uv.lock` 已同步 langchain-text-splitters 1.1.2。
- 分片参数实验：`rag-agent chunk-report docs/requirements.md --chunk-sizes 400,800,1200 --chunk-overlaps 0,80`。实测结果：

  | chunk_size / overlap | 分片数 | 中位数 | p90 | 最长 | 总字符 |
  |---|---|---|---|---|---|
  | 400 / 0 | 35 | 93 | 244 | 386 | 3908 |
  | 400 / 80 | 35 | 95 | 244 | 386 | 3970 |
  | 800 / 0 | 33 | 95 | 244 | 474 | 3911 |
  | 800 / 80 | 33 | 95 | 244 | 474 | 3911 |
  | 1200 / 0 | 33 | 95 | 244 | 474 | 3911 |
  | 1200 / 80 | 33 | 95 | 244 | 474 | 3911 |

- 同一文档的结构实测：`document chars` 3975，`section count` 33，section 长度最小 10、中位 95、最大 474，其中 32 个 section 不足 400 字符，没有任何 section 达到 800 字符，33 个 section 全部带标题。
- 实验结论：本文档上 `chunk_size ≥ 474` 后分片数恒为 33，`chunk_overlap` 只在同一 section 被切成多片时才起作用，因此 800/1200 两档与不同 overlap 的数字完全相同。**chunk_size 不是唯一旋钮，文档结构（标题密度）同样决定结果**；对无标题的 TXT/PDF，chunk_size 才会成为主要控制项。
- 待评测集验证的可选改进（阶段 8 再决定，现在不凭感觉调参）：是否允许同一标题路径下的相邻 section 合并到 chunk_size；是否过滤或前向合并过短分片（本文档最短分片 10 字符，35 片中有 30 片不足 200 字符）。

阶段 5 证据（2026-09-10 实际执行）：

- 代码提交：`cb9a25f`。
- `ruff check`：All checks passed；`ruff format --check`：51 files already formatted。
- `mypy`（strict，files = ["src"]）：Success: no issues found in 27 source files。
- `pytest`：157 passed, 2 skipped（在 DSH 沙箱内执行）；另有 18 个使用 `tmp_path` 的用例受沙箱限制无法在此运行，由学习者在本地确认，两处合计 175 passed, 2 skipped。
- 真实资料：`data/raw/` 中放入两份扫地机器人说明书 PDF（该目录被 .gitignore 忽略，不进版本库）。
- 手册 A（`20220726150404543.pdf`）：32 页、13578 字符、37 个分片、覆盖 29 页；分片长度最小 27、中位 240、p90 785、最大 799；入库需要 4 次 Embedding 请求（每批 10 个分片）。
- 手册 B（`20250905161902236.pdf`）：36 页但取不到任何文本。诊断结论是**文字被转成矢量轮廓**：页面资源里没有字体对象，内容流中 `BT`/`Tj`/`TJ` 出现次数均为 0，只有路径绘制指令。加载器给出 `empty_document`，这是正确行为；要使用这份手册需要 OCR，超出当前范围。
- 五步真实验收结果：

  1. `rag-agent ingest data\raw`：手册 A `indexed`、`chunk_count` 37、`vector_count` 37、`document_id` `doc-bf97be1e795f4f2a`；手册 B `failed`、`empty_document`；整体 `status` 为 `partial`，退出码 1（有文件失败的设计行为）。
  2. 再次执行同一命令：手册 A 变为 `unchanged`、仍为 37 个分片，日志中没有新的 `indexed` 记录，即**没有产生任何 Embedding 调用**。
  3. 落库状态：`documents` 一行（`pages 32`、`chunks 37`、`version 4bef75f516aa`），`ingestion_jobs` 依次为 `succeeded`、`skipped`、`failed`、`failed`，向量总数 37。
  4. `rag-agent delete-document 20220726150404543.pdf`：`removed`、`vector_count` 37、`status` 为 `ok`，向量与元数据一并清除。
  5. `rag-agent reindex`：手册 A 重新 `indexed`（37 个分片），手册 B 仍失败，退出码 1。
- 已知限制：pypdf 会为手册 A 的 CFF Type1 字体输出 `fontTools is required` 警告（每次加载 3 条），说明当前使用的是回退编码解析；文本能正确取出，但计划安装 `fonttools` 后重新加载并对比字符数与分片数，用数据确认提取是否更完整。

已知环境注意事项：

- 在 DSH 沙箱内运行 pytest 时，pytest 创建目录用的 `tempfile.mkdtemp()`（`.venv/Lib/site-packages/_pytest/cacheprovider.py:66`）和 `mkdir(mode=0o700)`（`.venv/Lib/site-packages/_pytest/pathlib.py:232`）都会产生沙箱进程之后无法访问的目录，表现为 `PytestCacheWarning` 或使用 `tmp_path` 的用例报 `PermissionError`，并遗留 `pytest-cache-files-*` 或 `.pytest_tmp` 目录。这是沙箱副作用；在普通终端运行 pytest 不受影响。

## 当前已实现能力

- Python 3.13 src 工程布局。
- uv 依赖锁定。
- 类型化环境配置。
- 项目根目录相对路径转绝对路径。
- JSON 结构化日志。
- 环境与核心依赖健康检查。
- 可复现的环境重建说明（Python 版本、uv sync、健康检查和质量工具命令）。
- 供应商无关的聊天与 Embedding Provider 协议，以及通义千问适配器。
- 模型调用的超时、重试、错误分类、耗时和 token 用量记录。
- 无网络、无密钥即可运行的 Fake Chat 与 Fake Embedding。
- 由 Settings 组装的模型实例，以及最小 Prompt -> Model -> Parser 问答链路。
- `rag-agent ask "问题"` 命令行问答，输出结构化 JSON。
- TXT、Markdown 和 PDF 文档加载，PDF 保留真实页码。
- 编码识别与文本规范化，规范化结果幂等。
- 稳定文档 ID、内容校验值与版本，可追溯每个文档的来源。
- 批量加载的失败隔离、重复内容标记与目录枚举。
- 标题感知的递归字符分片，保留标题层级路径、页码与跨页字符范围，并强制分片可回溯到原文。
- `rag-agent chunk-report` 命令行参数实验，输出分片数量与长度分布。
- `rag-agent ingest <路径>` 命令行入库，支持单文件与目录，逐文件隔离失败。
- SQLite 记录文档当前状态、内容版本历史与每次入库任务（成功、跳过、失败与错误码）。
- Chroma 持久化向量索引，按稳定 chunk ID 幂等写入，内容变更后自动清除旧分片。
- 未变更文档重新入库时短路，不重复调用 Embedding。
- `rag-agent reindex` 重建全部索引并清除已删除文件的残留向量。
- `rag-agent delete-document <来源>` 同时删除向量与元数据。

## 尚未实现

- 语义检索、低置信度拒答和来源引用（阶段 6 起）。
- 用户、设备、订单和工单工具。
- LangGraph 路由、Checkpoint 和人工确认。
- RAG 离线评测、Streamlit 界面、安全降级和 Docker 交付。
- 矢量轮廓 PDF 的文本提取（需要 OCR，当前明确不支持）。

## 最近结构变化

本次同步（阶段 5）相对上一次的主要变化：

- 新增 `src/rag_agent/storage/`，承载 SQLite 元数据与任务记录。
- 新增 `src/rag_agent/vectorstore/`，封装 Chroma 向量读写。
- 新增 `src/rag_agent/ingestion/pipeline.py`，串起加载、分片、Embedding 与两存储写入。
- `src/rag_agent/__main__.py` 新增 `ingest`、`reindex`、`delete-document` 与 `--force`。
- `tests/unit/` 新增四个测试文件；`ingestion_test_support.py` 增加隔离向量存储构造。
