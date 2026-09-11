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
├── data/
│   ├── eval/
│   │   ├── qa_set.jsonl          # 评测集，随代码提交
│   │   └── reports/              # 评测报告，随代码提交作为基线记录
│   ├── raw/                      # 原始资料，本地数据，不提交
│   ├── chroma/                   # 向量索引，本地数据，不提交
│   └── rag_agent.sqlite3         # 元数据，本地数据，不提交
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
│       │   ├── citation.py
│       │   ├── documents.py
│       │   ├── errors.py
│       │   └── retrieval.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── metrics.py
│       │   ├── report.py
│       │   └── runner.py
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── minimal_qa.py
│       │   └── rag_answer.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── filters.py
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
│       ├── retrieval/
│       │   ├── __init__.py
│       │   └── retriever.py
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
        ├── test_cli_answer.py
        ├── test_cli_ask.py
        ├── test_cli_chunk_report.py
        ├── test_cli_ingest.py
        ├── test_cli_search.py
        ├── test_cli_evaluate.py
        ├── test_documents.py
        ├── test_eval_dataset.py
        ├── test_eval_metrics.py
        ├── test_eval_runner.py
        ├── test_fake_providers.py
        ├── test_health.py
        ├── test_ingest_pipeline.py
        ├── test_ingestion_filters.py
        ├── test_loaders.py
        ├── test_loaders_pdf.py
        ├── test_logging.py
        ├── test_minimal_chain.py
        ├── test_normalize.py
        ├── test_provider_errors.py
        ├── test_provider_factory.py
        ├── test_qwen_adapter.py
        ├── test_qwen_stream.py
        ├── test_rag_answer.py
        ├── test_settings.py
        ├── test_sqlite_store.py
        └── test_splitters.py
~~~

.git、.venv、缓存、字节码和本地运行数据不在结构图中显示；`data/eval/` 属于随代码提交的内容，因此单独列出。

## 根目录文件

| 文件 | 职责 | 当前作用 |
|---|---|---|
| .env.example | 声明允许使用的环境变量名称 | 提供配置模板，含路径、模型、重试与检索参数，不包含真实密钥 |
| .gitignore | 定义 Git 排除规则 | 防止提交密钥、虚拟环境、缓存、日志和本地数据库 |
| PROJECT_STRUCTURE.md | 保存真实项目结构和职责说明 | 用户要求同步结构时由 Codex 更新 |
| README.md | 说明项目定位、可复现的环境重建步骤、当前能力和边界 | 不保存阶段操作流程 |
| pyproject.toml | 定义项目、Python 范围、直接依赖和工具配置 | uv、pytest、Ruff 和 mypy 的共同配置入口 |
| uv.lock | 锁定完整依赖图 | 保证不同环境安装相同版本 |

## 源码

| 文件 | 职责 | 涉及知识 |
|---|---|---|
| src/rag_agent/__init__.py | 声明 rag_agent Python 包并提供版本号 | 包、模块、包元数据 |
| src/rag_agent/__main__.py | 提供 `health`、`ask`、`answer`、`search`、`chunk-report`、`ingest`、`reindex`、`delete-document` 八个命令，退出码 0 成功、1 部分失败或拒答、2 错误 | 命令行参数、进程退出码、错误到退出码的映射 |
| src/rag_agent/health.py | 集中检查 Python、虚拟环境、依赖和核心模块导入 | 运行环境、依赖元数据、模块导入 |
| src/rag_agent/config/__init__.py | 对外暴露配置与 Provider 组装接口 | 包的公共 API |
| src/rag_agent/config/settings.py | 从环境变量读取配置并解析项目绝对路径，含模型名、超时、重试和检索参数 | Pydantic、环境配置、路径稳定性 |
| src/rag_agent/config/providers.py | 依据 Settings 组装通义千问聊天与 Embedding 实例，缺密钥时在发请求前失败 | 依赖注入、组装与配置边界 |
| src/rag_agent/domain/__init__.py | 对外暴露领域模型与摄取错误 | 包的公共 API |
| src/rag_agent/domain/documents.py | 定义 FileType、DocumentStatus、DocumentPage、SourceDocument、DocumentChunk，以及稳定文档 ID、校验值和版本派生 | 领域模型、稳定标识、内容版本 |
| src/rag_agent/domain/errors.py | 定义八类带稳定 code 的摄取错误，含分片错误 | 错误隔离、异常设计 |
| src/rag_agent/domain/retrieval.py | 定义 RetrievalHit、RetrievalResult 与置信判定，reason 解释判定依据，margin 记录前两名分差供评测分析 | 领域模型、置信判定、可解释性 |
| src/rag_agent/domain/citation.py | 定义 Citation 与摘录压缩，label 给出可读定位 | 领域模型、来源追溯 |
| src/rag_agent/ingestion/__init__.py | 对外暴露加载、规范化、分片、统计与入库接口 | 模块边界 |
| src/rag_agent/ingestion/normalize.py | 识别编码并规范化文本：BOM 嗅探、编码回退、换行与控制字符处理、空行折叠、幂等保证 | 字符编码、文本规范化、幂等性 |
| src/rag_agent/ingestion/loaders.py | 把 TXT、Markdown 和 PDF 加载为 SourceDocument，批量失败隔离、重复内容标记与目录枚举 | Document Loader、错误隔离、批量一致性 |
| src/rag_agent/ingestion/splitters.py | 标题感知与递归字符分片：按标题切 section、保留标题层级路径、计算跨页字符范围、强制分片可回溯到原文 | Text Splitter、分片参数、字符偏移、不变量校验 |
| src/rag_agent/ingestion/stats.py | 汇总分片数量与长度分布，含最近秩分位数和直方图，供参数实验比较 | 分布统计、分位数、实验可重复性 |
| src/rag_agent/ingestion/pipeline.py | 串起加载、分片、分批 Embedding、写向量与写元数据；未变更短路、内容变更替换旧分片、入库前过滤目录类分片、失败记录任务且保留旧版本 | 幂等性、增量更新、两存储一致性、噪声过滤 |
| src/rag_agent/ingestion/filters.py | 用点引导线比例识别目录或索引类分片，规则确定且可关闭，供阶段 8 做 A/B | 启发式规则、检索噪声、可实验性 |
| src/rag_agent/generation/__init__.py | 对外暴露最小问答链路接口 | 模块边界 |
| src/rag_agent/generation/minimal_qa.py | 用 LCEL 组装 Prompt -> Model -> Parser，并返回带模型证据的结构化答案 | LCEL、Runnable 协议、结构化输出 |
| src/rag_agent/generation/rag_answer.py | 编号上下文、依据约束提示词、引用校验、六类拒答原因；拆出 prepare/finalize 供流式复用 | Grounding、结构化输出、引用校验、幻觉来源 |
| src/rag_agent/observability/__init__.py | 对外暴露日志配置接口 | 模块边界 |
| src/rag_agent/observability/logging.py | 输出 JSON 结构化日志，并把 httpx 的 INFO 日志降为 WARNING | Python logging、结构化数据、异常记录 |
| src/rag_agent/providers/__init__.py | 对外暴露协议、错误类型、真实实现和 Fake 实现 | 模块边界、公共 API |
| src/rag_agent/providers/base.py | 定义 ChatMessage、ChatResponse、EmbeddingResponse 等值对象和 ChatModel、EmbeddingModel 协议，以及七类模型错误 | 依赖倒置、结构化类型、错误分类 |
| src/rag_agent/providers/qwen.py | 通过 httpx 调用 DashScope OpenAI 兼容端点，实现超时、指数退避重试、状态码分类、耗时与 token 记录 | HTTP 客户端、重试策略、密钥外置 |
| src/rag_agent/providers/fake.py | 提供脚本化 FakeChatModel 和确定性 FakeEmbeddingModel | 测试替身、确定性测试 |
| src/rag_agent/storage/__init__.py | 对外暴露元数据仓储接口 | 模块边界 |
| src/rag_agent/storage/sqlite.py | SQLite 元数据存储：documents、document_versions、ingestion_jobs 三张表，按来源 upsert、查询、删除并记录任务 | sqlite3、事务、唯一约束、版本历史 |
| src/rag_agent/vectorstore/__init__.py | 对外暴露向量存储与匹配结果接口 | 模块边界 |
| src/rag_agent/vectorstore/chroma.py | Chroma 封装：按稳定 chunk ID 幂等 upsert、按文档取 ID 与内容、删除、向量查询并重建 DocumentChunk，支持按来源过滤 | 向量维度、距离度量、幂等写入、元数据回读 |
| src/rag_agent/retrieval/__init__.py | 对外暴露检索器接口 | 模块边界 |
| src/rag_agent/retrieval/retriever.py | 查询向量化、Top-K、来源过滤与阈值判定；单次调用可覆盖 top_k、threshold、source | 语义相似度、Top-K、阈值取舍 |

| data/eval/qa_set.jsonl | 49 条评测用例（43 可回答、6 不可回答），含预期页码与参考答案 | 评测集设计、标注一致性 |
| data/eval/reports/ | 每次评测生成的 Markdown 与 JSON 报告 | 基线记录、可复现性 |
| src/rag_agent/evaluation/dataset.py | 加载并校验评测集：必填字段、重复 ID、可回答与页码的一致性 | 数据契约、校验 |
| src/rag_agent/evaluation/metrics.py | Recall@K、MRR、引用正确率、拒答正确率、决策正确率与忠实度代理指标 | 检索评测与生成评测的区别 |
| src/rag_agent/evaluation/runner.py | 检索模式与回答模式两种跑法，逐条记录并可汇总 | 实验条件记录 |
| src/rag_agent/evaluation/report.py | 输出 Markdown 与 JSON 报告，附「需要关注的用例」小节 | 回归对比、可读性 |

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
| tests/unit/test_retriever.py | 精确内容命中排第一、分数降序与排名、top-k 默认与覆盖、阈值高低分支、空索引、来源过滤、非法配置与调用参数 |
| tests/unit/test_cli_search.py | search 的用法错误、命中输出、低置信度退出码 1、空索引与阈值越界 |
| tests/unit/test_rag_answer.py | 引用校验、编造编号被丢弃与上报、六类拒答原因、上下文预算、提示词与 Provider 异常传播 |
| tests/unit/test_cli_answer.py | answer 的引用输出、拒答退出码、空索引不调用模型、`--stream` 先流后结果与缺密钥 |
| tests/unit/test_qwen_stream.py | SSE 增量顺序、噪声与 `[DONE]` 忽略、状态码分类、首个增量后不重试、Fake 分片 |
| tests/unit/test_ingestion_filters.py | 目录页识别、正常段落与标准表格不被误删、比例与最小行数可配置 |
| tests/unit/test_eval_dataset.py | 评测集解析与校验：必填字段、重复 ID、不可回答不得带页码、随包数据集规模与可回答比例 |
| tests/unit/test_eval_metrics.py | 命中排名与倒序排名、任意预期页匹配、空结果、忠实度代理、两种模式的汇总与决策正确率 |
| tests/unit/test_eval_runner.py | 检索与回答两种模式跑通，引用正确性、拒答统计、无命中不调用模型与报告序列化 |
| tests/unit/test_cli_evaluate.py | evaluate 命令的模式、limit、数据集错误、报告写出与缺密钥分支 |
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

阶段 6（检索器 V1 与可解释结果）已完成并通过验收，含真实检索实测与一次重要的负面发现。

阶段 7（RAG 回答与来源引用）已完成并通过验收，含真实带引用回答、真实 Token 流式输出，以及阶段 6 遗留问题的处置。

阶段 8（RAG 评测基线）已完成并通过验收，含 49 条评测集、检索与回答两条基线，以及一次用数据驱动的提示词改进。

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
- 字体解析对比实验：pypdf 原本为手册 A 的 CFF Type1 字体输出 3 条 `fontTools is required` 警告。加入 `fonttools` 4.64.0 后重新加载，**警告消失，提取结果完全不变**（13578 字符、32 页、37 个分片、分片长度最小 27、中位 240、最大 799），说明此前的回退编码解析已经准确，因此不需要重新入库。

阶段 6 证据（2026-09-10 实际执行）：

- 代码提交：`5164c47`。
- `ruff check`：All checks passed；`ruff format --check`：56 files already formatted。
- `mypy`（strict，files = ["src"]）：Success: no issues found in 30 source files。
- `pytest`：181 passed, 2 skipped（在 DSH 沙箱内执行）；另有 18 个使用 `tmp_path` 的用例受沙箱限制，由学习者在本地确认，两处合计 199 passed, 2 skipped。
- 真实检索实测（37 个分片的真实索引，每次查询 1 次 Embedding 请求）：

  | 查询 | best_score | rank 1 命中 | confident | 判定 |
  |---|---|---|---|---|
  | 制造商地址在哪里 | 0.5889 | `c0037` 第 32 页，制造商名称与地址 | true | 正确 |
  | 使用产品前需要先做什么 | 0.6318 | `c0004` 第 3 页，**目录页** | true | 命中目录而非正文 |
  | 这台扫地机器人明年会涨价吗 | 0.5897 | `c0030` 第 24 页，故障排查表 | true | 不应置信 |
  | 同上，`--threshold 0.5` | 0.5897 | 同上 | true | 仍置信 |
  | 执行标准，`--top-k 2 --source` | 0.5282 | `c0036` 第 28 页，执行标准 | true | 正确，来源过滤生效 |

- **发现一：绝对余弦阈值不能区分「该答」与「不该答」。** 正确答案的 best_score 是 0.5889，知识库中完全不存在的问题（涨价预测）best_score 是 0.5897，两者几乎相同；把阈值提到 0.5 仍无法拒答，而继续提高到 0.6 会把正确答案一并拒掉。因此 `retrieval_threshold` 的默认 0.35 只是**松下界**，用于挡掉接近 0 的匹配，**不构成拒答机制**。这一结论直接决定阶段 7 必须依靠「只能依据 Context 回答」的提示词与引用校验，而不是分数阈值。
- **发现二：目录页是检索噪声。** 第二个查询的 rank 1 是目录分片（0.6318），真正含答案的分片以 0.6307 排第二，仅差 0.0011。目录页与各类问题都有词汇重叠，是可复现的干扰源。
- **发现三：top1 与 top2 的间隔同样不可分。** 相关查询的间隔为 0.0011，不相关查询的间隔为 0.0123，间隔反而更小，因此不能用它替代绝对阈值。这是先算数据再设计的一个反例，避免了凭感觉引入无效判据。
- 五次运行的分数集中在 0.42 至 0.63 之间，说明 `text-embedding-v4` 的余弦相似度基线整体偏高，这解释了发现一。
- 待阶段 8 用评测集回答的问题：是否需要重排、是否过滤或降权目录类分片、阈值取多少或改用相对判据。三条查询不是评测，只是第一个数据点，因此本阶段刻意未调整任何默认参数。

阶段 7 证据（2026-09-10 / 09-11 实际执行）：

- 代码提交：`1940c9d`。
- `ruff check`：All checks passed；`ruff format --check`：63 files already formatted。
- `mypy`（strict，files = ["src"]）：Success: no issues found in 33 source files。
- `pytest`：229 passed, 2 skipped（在 DSH 沙箱内执行）；另有 18 个使用 `tmp_path` 的用例受沙箱限制，由学习者在本地确认，两处合计 247 passed, 2 skipped。
- 真实带引用回答（学习者在本地执行）：

  | 提问 | 结果 | 引用 |
  |---|---|---|
  | 制造商地址在哪里 | answered，`status` 为 `ok` | `[1]` 第 32 页，答案即「苏州市吴中区郭巷街道淞苇路 518 号」 |
  | 产品型号是什么，售后电话是多少 | answered | `[2]` 第 27 页，答案「DBX23 / 400-886-8888」 |
  | 这台扫地机器人明年会涨价吗 | **refused**，`refusal_cause` 为 `model_insufficient` | 无，理由指出所有资料均未涉及价格与定价政策 |

- 第三问正是阶段 6 阈值失效的同一道题：检索 `confident` 仍为 true（best 0.5897），但回答层正确拒答。**这是「依据约束 + 引用校验」优于分数阈值的直接证据。**
- 第二问还暴露了一个有意思的现象：检索 rank 1 是目录分片（0.5664），真正含型号与电话的分片排第 2（0.5617），但模型**引用了 rank 2**，答案准确——生成阶段部分补偿了检索排序缺陷。
- 真实 Token 流式输出（本机实测）：13 个增量，首个增量 1555 ms 到达、末个 2791 ms、总计 2813 ms，说明是真实增量而非一次性缓冲；流式结果与一次性调用得到同样的引用（第 32 页）。
- 已知偏差：第一问的答案正文里没有 `[1]` 行内标注，编号只出现在结构化的 `citations` 字段中。这满足计划里「结构化答案 + 引用列表」的验收，但提示词第 2 条要求的行内标注并未被强制；是否补强制校验留给阶段 8 作为评测检查项。

阶段 6 遗留问题的处置（2026-09-11 实际执行）：

- **目录类分片：采用确定性过滤规则。** 新增 `ingestion/filters.py`，以「点引导线结尾带页码的行占比」识别目录或索引页（默认比例 0.3、最少 3 行），入库前过滤，并可用 `--keep-index-chunks` 关闭，供阶段 8 做 A/B。选择过滤而非降权的理由：清单页本身不含可回答的事实，且规则不需要人为设定权重。
- 实测效果：手册 A 由 **37 个分片降为 34 个**，恰好丢掉此前定位到的三个目录分片；对查询「使用产品前需要先做什么」，rank 1 由目录分片变为真正含答案的分片，**top1 与 top2 的间隔由 0.0011 提升到 0.0375**（约 34 倍）。这是可测量、可复现的改善，而不是主观判断。
- 过滤后的回答链路复核：同一问题 `answer` 仍为 `answered`，引用第 2 页（此前被目录页挤到第二名的分片），耗时 4253 ms。
- **重排：暂不引入。** 理由是没有评测集就无法证明重排带来收益，而重排会引入额外模型调用成本；阶段 8 用 Recall@K 与 MRR 判定是否需要。
- **阈值：角色重新定位为成本控制，而非正确性判定。** `is_confident` 的文档字符串已写明这一点，`RetrievalResult.margin` 新增为结构化字段，供阶段 8 分析相对信号是否有用（阶段 6 已证明它在这三条查询上不可分，因此本阶段不据此改判）。

阶段 8 证据（2026-09-11 实际执行）：

- 代码提交：`fe19ecd`。
- `ruff check`：All checks passed；`mypy`（strict，files = ["src"]）：Success: no issues found in 38 source files。
- `pytest`：268 passed, 2 skipped（在 DSH 沙箱内执行）；另有 19 个使用 `tmp_path` 的用例受沙箱限制，由学习者在本地确认，两处合计 287 passed, 2 skipped。
- 评测集：49 条（43 条可回答、6 条不可回答），覆盖 24 个页面、7 个类别，位于 `data/eval/qa_set.jsonl`，随代码提交。
- **检索基线**（`rag-agent evaluate`，确定性可重复）：

  | 指标 | 数值 |
  |---|---|
  | Recall@5 | 0.9535（43 条中命中 41 条） |
  | MRR | 0.8205 |
  | 平均最高相似度 | 0.6585 |
  | 平均前两名分差 | 0.0613 |

- **两条漏检的根因**：q01（潮湿地面能否使用）与 q03（儿童能否使用）的预期页都是第 2 页，但 top-5 完全没有第 2 页分片。原因是第 2 页的两个分片有 750 至 777 字符、塞进了 17 条与 12 条互不相关的规则，主题被稀释，打不过聚焦单一主题的短分片。**这两条的 best_score 分别是 0.6731 与 0.6429，都高于可回答组的中位数 0.6495，因此任何阈值都救不了它们。**
- **阶段 6 遗留问题的最终答案（用 49 条数据而非 3 条）**：可回答组的最高相似度区间是 [0.4938, 0.8689]，不可回答组是 [0.4415, 0.6020]，**两组重叠 0.1082**；前两名分差同样重叠（可回答 [0.0010, 0.2353]，不可回答 [0.0040, 0.0417]）。结论：**不存在能把两组分开的单一阈值或单一分差阈值**，阈值只能作为成本控制的下界，拒答必须由依据约束与引用校验承担。
- **回答基线**（指标口径修正后）：回答率 0.6744、引用正确率 0.8276、拒答正确率 1.0、决策正确率 0.7143、忠实度代理 0.5706；拒答原因分布为 `model_insufficient` 9、`unparseable` 5、`no_valid_citation` 6。
- **根因定位**：`unparseable` 全部是模型把 citations 写成了 `[3][9][11][2]` 这种非法 JSON；`no_valid_citation` 集中在故障排查类，模型引用的是说明书表格里的“序号”（8、9、13、14、19、22），而上下文只有 5 条资料，编号越界后被丢弃。
- **一次数据驱动的改进**：在提示词中明确 citations 必须是整数数组、并说明编号含义。重测结果——回答率 **0.6744 → 0.7907**（+11.6 个百分点），决策正确率 **0.7143 → 0.8163**，`unparseable` **5 → 0**，拒答正确率保持 1.0，引用正确率 0.8276 → 0.7941（回答变多、精度略降）。这是一次有前后对照的改进，而不是主观断言。
- **尚未解决**：`no_valid_citation` 仍为 6（表格序号混淆未被那条规则消除）；`model_insufficient` 为 9，其中 2 条由检索漏检导致，1 条（q30）是检索命中但生成没有答出来。
- **指标口径说明**：检索按页粒度计分（评测集能可靠标注到页，未做分片级人工标注）；忠实度是代理指标，计算答案的 4-gram 在**引用分片全文**中的覆盖率，可被照抄误导、也无法识别“用对的原文得出错结论”，因此必须与引用正确率、拒答正确率一起看。
- **运行间波动**：检索指标确定可重复；回答指标会波动（两次基线运行的 `unparseable` 分别为 4 与 5），因此回答模式的对比不能只看单次结果。
- 可重复执行：`rag-agent evaluate --mode retrieval`（仅查询 Embedding）与 `rag-agent evaluate --mode answer`（每条约 2 个请求），报告写入 `data/eval/reports/`。

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
- Top-K 语义检索，命中项带分片内容、来源、页码、标题、字符范围、相似度与排名。
- 检索可按来源过滤，并支持在命令行覆盖 top_k、threshold 与 source。
- 置信判定与判定依据说明（`confident` 与 `reason`），低置信度以退出码 1 表达而非崩溃。
- `rag-agent search "问题"` 检索调试命令，输出可解释的命中列表。
- 依据检索结果生成带引用校验的答案：只能依据编号资料作答，编造的引用会被丢弃并上报，没有可核验引用时整条回答被拒。
- 六类机器可读的拒答原因，便于统计拒答正确率。
- 检索为空时不调用聊天模型，直接拒答并省下一次请求。
- `rag-agent answer "问题"` 命令行问答，支持 `--stream` 真实 Token 增量输出。
- Provider 层的 SSE 流式解析，首个增量之后不再重试，避免重复输出。
- 入库前过滤目录或索引类分片，规则确定、可关闭。
- 49 条带预期页码与参考答案的离线评测集，覆盖安全、使用、保养、故障、参数、合规与范围外问题。
- `rag-agent evaluate --mode retrieval|answer` 一条命令生成 Markdown 与 JSON 报告，并列出需要关注的用例。
- 检索指标（Recall@K、MRR）与回答指标（回答率、引用正确率、拒答正确率、决策正确率、忠实度代理）分开度量。

## 尚未实现

- 用户、设备、订单和工单工具。
- LangGraph 路由、Checkpoint 和人工确认。
- Streamlit 界面、安全降级和 Docker 交付。
- 矢量轮廓 PDF 的文本提取（需要 OCR，当前明确不支持）。
- 重排：已确认阈值与分差都不可分，是否需要重排需靠 Recall@K 与 MRR 的进一步实验判断。
- 长分片的主题稀释问题（第 2 页 750 字符分片导致两条安全类问题漏检），需要按编号条目二次切分的实验。
- 表格序号与上下文编号混淆导致的引用越界（6 条），需要更强的格式约束或校验提示。

## 最近结构变化

本次同步（阶段 8）相对上一次的主要变化：

- 新增 `src/rag_agent/evaluation/`，承载评测集、指标、执行器与报告。
- 新增 `data/eval/qa_set.jsonl` 与 `data/eval/reports/`，并首次把评测内容纳入版本库。
- `src/rag_agent/__main__.py` 新增 `evaluate` 命令与 `--mode`、`--dataset`、`--out`、`--limit`。
- `src/rag_agent/generation/rag_answer.py` 的提示词明确 citations 为整数数组并说明编号含义。
- `tests/unit/` 新增四个测试文件。
