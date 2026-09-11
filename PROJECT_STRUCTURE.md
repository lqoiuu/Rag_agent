# 项目结构与文件职责

最后同步：2026-09-11
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
│   ├── business/
│   │   └── seed.json             # 模拟用户、设备、订单，随代码提交
│   ├── eval/
│   │   ├── qa_set.jsonl          # 评测集，随代码提交
│   │   └── reports/              # 评测报告，随代码提交作为基线记录
│   ├── raw/                      # 原始资料，本地数据，不提交
│   ├── chroma/                   # 向量索引，本地数据，不提交
│   └── rag_agent.sqlite3         # 元数据与业务数据，本地数据，不提交
├── src/
│   └── rag_agent/
│       ├── __init__.py
│       ├── __main__.py
│       ├── health.py
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── devices.py
│       │   ├── graph.py
│       │   ├── intent.py
│       │   ├── nodes.py
│       │   ├── routing.py
│       │   └── state.py
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
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── checkpoints.py
│       │   ├── conversation.py
│       │   └── schema.py
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
│       │   ├── business.py
│       │   └── sqlite.py
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── business.py
│       │   ├── errors.py
│       │   └── models.py
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── app.py                  # 界面入口与路由；页面本身在 app_pages/
│       │   ├── services.py             # 进程级共享资源与路径助手
│       │   └── app_pages/
│       │       ├── __init__.py
│       │       ├── chat.py
│       │       ├── knowledge.py
│       │       └── search.py
│       └── vectorstore/
│           ├── __init__.py
│           └── chroma.py
└── tests/
    ├── conftest.py                 # 把每个用例的临时目录放在项目内的 .pytest_tmp/
    ├── integration/
    │   └── test_qwen_live.py
    └── unit/
        ├── agent_test_support.py
        ├── ingestion_test_support.py
        ├── memory_test_support.py
        ├── model_test_support.py
        ├── pdf_fixtures.py
        ├── test_agent_graph.py
        ├── test_agent_intent.py
        ├── test_agent_memory.py
        ├── test_agent_routing.py
        ├── test_business_repository.py
        ├── test_chroma_store.py
        ├── test_chunk_stats.py
        ├── test_cli_agent.py
        ├── test_cli_answer.py
        ├── test_cli_ask.py
        ├── test_cli_chat.py
        ├── test_cli_chunk_report.py
        ├── test_cli_ingest.py
        ├── test_cli_search.py
        ├── test_cli_tool.py
        ├── test_cli_evaluate.py
        ├── test_conversation_memory.py
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
        ├── test_memory_checkpoints.py
        ├── test_minimal_chain.py
        ├── test_normalize.py
        ├── test_provider_errors.py
        ├── test_provider_factory.py
        ├── test_qwen_adapter.py
        ├── test_qwen_stream.py
        ├── test_rag_answer.py
        ├── test_rag_answer_stream_context.py
        ├── test_settings.py
        ├── test_ui_app.py
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
| src/rag_agent/__main__.py | 提供 `health`、`ask`、`answer`、`search`、`chunk-report`、`ingest`、`reindex`、`delete-document`、`evaluate`、`tool`、`agent`、`chat`、`thread` 十三个命令，退出码 0 成功、1 部分失败或需用户动作、2 错误 | 命令行参数、进程退出码、错误到退出码的映射 |
| src/rag_agent/health.py | 集中检查 Python、虚拟环境、依赖和核心模块导入 | 运行环境、依赖元数据、模块导入 |
| src/rag_agent/agent/__init__.py | 对外暴露状态、意图、节点、图与运行结果 | 模块边界 |
| src/rag_agent/agent/state.py | AgentState：输入、多轮上下文字段、意图、澄清轮次、知识与工具结果、工单流程、状态与节点轨迹，列表字段带 reducer | State 设计、Reducer、状态可观测性 |
| src/rag_agent/agent/intent.py | 模型辅助的意图识别，可接收最近对话作为上下文；解析失败或置信度过低一律回退 unknown | 结构化输出、确定性回退 |
| src/rag_agent/agent/nodes.py | 八个节点：意图识别、知识问答、设备查询、工单信息收集、待确认、创建工单、澄清、失败；创建节点以 `interrupt()` 作为第一条语句，暂停早于任何工具调用 | Node 职责单一、写操作双重屏障 |
| src/rag_agent/agent/devices.py | 设备号正则与解析顺序（调用方参数 → 当前消息 → 对话窗口）的唯一一份定义，供设备节点与路由规则共用 | 确定性抽取、规则单一来源 |
| src/rag_agent/agent/routing.py | 在模型判定之后应用的确定性改判规则：`prefer_device` 与两张词表，纯函数、只向 device 收敛 | 确定性路由、模型判断与代码判断的边界 |
| src/rag_agent/agent/graph.py | 条件边路由、澄清计数与 recursion_limit 守卫、Checkpointer 挂载、`Command(resume=...)` 恢复入口、AgentRun 与 ChatTurn 结果视图 | Edge、条件边、Interrupt、循环限制 |
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
| src/rag_agent/generation/rag_answer.py | 编号上下文、依据约束提示词、引用校验、六类拒答原因；拆出 prepare/finalize 供流式复用；可把最近对话前置到用户消息 | Grounding、结构化输出、引用校验、幻觉来源 |
| src/rag_agent/memory/__init__.py | 对外暴露 Checkpointer、会话登记表、窗口与偏好接口 | 模块边界 |
| src/rag_agent/memory/schema.py | Checkpoint 三张表与 `_threads`、`user_preferences` 的建表语句，供检查点与对话登记共同使用 | SQLite 表设计、主键与索引 |
| src/rag_agent/memory/checkpoints.py | 继承 BaseCheckpointSaver 的 SQLite 实现：按 channel 版本存 blob、写检查点与父指针、按任务写 writes、回溯父链、按线程删除；用可重入锁串行化事务 | Checkpoint 结构、序列化、线程安全、事务 |
| src/rag_agent/memory/conversation.py | 会话归属校验、消息窗口裁剪与 Prompt 渲染、长期偏好读写 | 短期记忆与长期记忆的边界、上下文预算 |
| src/rag_agent/observability/__init__.py | 对外暴露日志配置接口 | 模块边界 |
| src/rag_agent/observability/logging.py | 输出 JSON 结构化日志，并把 httpx 的 INFO 日志降为 WARNING | Python logging、结构化数据、异常记录 |
| src/rag_agent/providers/__init__.py | 对外暴露协议、错误类型、真实实现和 Fake 实现 | 模块边界、公共 API |
| src/rag_agent/providers/base.py | 定义 ChatMessage、ChatResponse、EmbeddingResponse 等值对象和 ChatModel、EmbeddingModel 协议，以及七类模型错误 | 依赖倒置、结构化类型、错误分类 |
| src/rag_agent/providers/qwen.py | 通过 httpx 调用 DashScope OpenAI 兼容端点，实现超时、指数退避重试、状态码分类、耗时与 token 记录 | HTTP 客户端、重试策略、密钥外置 |
| src/rag_agent/providers/fake.py | 提供脚本化 FakeChatModel 和确定性 FakeEmbeddingModel | 测试替身、确定性测试 |
| src/rag_agent/storage/__init__.py | 对外暴露元数据仓储接口 | 模块边界 |
| src/rag_agent/storage/sqlite.py | SQLite 元数据存储：documents、document_versions、ingestion_jobs 三张表，按来源 upsert、查询、删除并记录任务；连接禁用线程绑定并用可重入锁串行化事务，文件库启用 WAL | sqlite3、事务、唯一约束、版本历史、线程安全 |
| src/rag_agent/vectorstore/__init__.py | 对外暴露向量存储与匹配结果接口 | 模块边界 |
| src/rag_agent/vectorstore/chroma.py | Chroma 封装：按稳定 chunk ID 幂等 upsert、按文档取 ID 与内容、删除、向量查询并重建 DocumentChunk，支持按来源过滤 | 向量维度、距离度量、幂等写入、元数据回读 |
| src/rag_agent/storage/business.py | 模拟用户、设备、订单、工单四张表与仓储接口，按 seed 文件幂等灌入，保修到期日按整数月计算；连接与事务处理同元数据存储 | SQLite、确定性数据、日期算术、线程安全 |
| src/rag_agent/tools/__init__.py | 对外暴露工具契约、错误类型与四个工具函数 | 模块边界 |
| src/rag_agent/tools/models.py | Pydantic 参数与结果模型，含工单草稿、确认标记与派生幂等键 | Schema 校验、幂等键设计 |
| src/rag_agent/tools/errors.py | 工具错误分类：not_found、permission_denied、invalid_argument、confirmation_required、conflict、unavailable | 工具边界、错误语义 |
| src/rag_agent/tools/business.py | 四个契约化工具：用户、设备、订单查询与工单创建；失败转成结构化结果而不是字符串 | Function Calling 契约、权限、幂等性与读写风险 |
| src/rag_agent/retrieval/__init__.py | 对外暴露检索器接口 | 模块边界 |
| src/rag_agent/retrieval/retriever.py | 查询向量化、Top-K、来源过滤与阈值判定；单次调用可覆盖 top_k、threshold、source | 语义相似度、Top-K、阈值取舍 |
| src/rag_agent/ui/__init__.py | 对外暴露资源容器与资源获取接口 | 模块边界 |
| src/rag_agent/ui/app.py | 界面入口与路由：三页 `st.Page` 导航、会话状态初始化、侧边栏身份与会话选择、把「界面消息」与「Checkpoint 状态」的差别写在界面上 | Streamlit 多页、`st.navigation`、session state 与资源缓存的职责划分 |
| src/rag_agent/ui/services.py | 进程级共享资源容器（向量库、模型、三个 SQLite 句柄、检索器），用 `st.cache_resource` 带 TTL 缓存并在释放时关闭句柄 | 缓存生命周期、依赖注入 |
| src/rag_agent/ui/app_pages/chat.py | 对话页：流式生成状态、引用卡片、工具调用步骤、待确认写操作的确认/取消按钮，并区分本轮回合与整条会话的轨迹 | 生成器流式、`st.chat_message`、写操作门禁的界面表达 |
| src/rag_agent/ui/app_pages/knowledge.py | 知识库页：上传落盘后入库、文档列表（分片与向量数）、删除、重建索引；每次 rerun 重读 SQLite，提交结果用一次性会话值传递 | 幂等入库、rerun 去重、文件上传 |
| src/rag_agent/ui/app_pages/search.py | 检索调试页：命中表、相似度、前两名分差、置信判定依据，并把「阈值不可分」的实测结论写在页面上 | 检索可解释性、调试界面 |
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
| tests/unit/agent_test_support.py | Agent 测试辅助：脚本化意图/回答/字段提取回复，以及内存向量库与模拟业务仓储 |
| tests/unit/memory_test_support.py | 会话记忆测试辅助：每次生成一个独立的共享缓存内存库名，使第二个连接仍能读到第一个连接写入的状态 |
| tests/unit/test_agent_intent.py | 意图标签识别、非法 JSON、未知标签、低置信度降级、阈值可配、非数值置信度 |
| tests/unit/test_agent_graph.py | 四类意图各走对路径、澄清分支、权限失败、待确认不写库、暂停后确认只写一次、澄清上限、**结构上不存在收集直达创建的边** |
| tests/unit/test_agent_memory.py | 第二轮读到第一轮消息、窗口裁剪而库中保留全量、线程之间不共享上下文、取消确认不写库、重复恢复不会二次写入、偏好进入 Prompt、从消息解析设备号、有历史时追问仍可用且轨迹留改判记录 |
| tests/unit/test_agent_routing.py | 改判规则的边界：指代式追问命中；模型已判 device、`ticket`、无指代、无设备号、含产品知识话题词五种情况都必须**不改判** |
| tests/unit/test_memory_checkpoints.py | 关闭后重新打开仍能读到会话、线程隔离、父链回溯与最新在前、按 checkpoint_id 精确取回、删除线程清理三张表、同一检查点重复写入幂等、limit 与 before 过滤 |
| tests/unit/test_conversation_memory.py | 窗口保留最新消息且丢弃非法记录、空上下文渲染为空串、上下文字块标注为「不作为事实依据」、会话归属校验、未绑定会话可被认领一次、偏好按用户隔离且不随会话清除、**登记表与检查点解析同一个数据库目标** |
| tests/unit/test_cli_chat.py | chat 生成或复用 thread、两次独立调用共享同一会话、跨用户被拒、待确认不写库、--cancel 不写库、--confirm 建单、thread list/clear/preferences、参数互斥与缺密钥 |
| tests/unit/test_cli_agent.py | agent 命令的轨迹输出、设备查询、待确认退出码 1、不再接受 `--confirm`、缺密钥 |
| tests/unit/test_business_repository.py | 模拟数据幂等灌入、类型化查询、保修状态随参考日期变化、工单幂等键唯一约束、整数月加法边界 |
| tests/unit/test_tools.py | 四个工具的契约、掩码字段、权限拒绝、确认要求、幂等重复调用、Schema 违规与存储故障可重试 |
| tests/unit/test_cli_tool.py | tool list 契约输出、查询成功、权限失败退出码 1、未确认拒写、重复创建返回同一工单、参数错误退出码 2 |
| tests/unit/test_rag_answer.py | 引用校验、编造编号被丢弃与上报、六类拒答原因、上下文预算、提示词与 Provider 异常传播 |
| tests/unit/test_rag_answer_stream_context.py | 流式与非流式构造同一条提示词、第一轮提示词保持不变、流式增量顺序与拼接结果 |
| tests/unit/test_ui_app.py | 用 Streamlit 官方 AppTest 无浏览器驱动界面（注入假模型，测试内不联网络）：入口与三页均无异常渲染、空索引有提示、侧边栏新建会话、设备问题走业务工具、报修暂停且取消不写库、确认只建一张工单、上传真的落盘并入库、检索页给出判定依据、流式开关走知识路径 |
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
| docs/adr/0003-conversation-memory.md | 会话持久化用自实现 SQLite Checkpointer、确认用节点内 Interrupt、窗口与长期偏好的边界 |
| docs/adr/0004-streamlit-ui.md | 界面不承载业务规则、状态分两层、流式与引用校验并存、本轮回合与累积视图的区分 |

## 当前阶段

阶段 1（Python 3.13 工程骨架与环境）已完成并通过验收。

阶段 2（模型适配层与最小问答）已完成并通过验收，含真实模型联网验证。

阶段 3（文档加载与规范化）已完成并通过验收。

阶段 4（文本清洗、分片与参数实验）已完成并通过验收。

阶段 5（向量化、持久化与增量索引）已完成并通过验收，含真实说明书入库验证。

阶段 6（检索器 V1 与可解释结果）已完成并通过验收，含真实检索实测与一次重要的负面发现。

阶段 7（RAG 回答与来源引用）已完成并通过验收，含真实带引用回答、真实 Token 流式输出，以及阶段 6 遗留问题的处置。

阶段 8（RAG 评测基线）已完成并通过验收，含 49 条评测集、检索与回答两条基线，以及一次用数据驱动的提示词改进。

阶段 9（业务工具与工具契约）已完成并通过验收，含模拟业务数据、四个契约化工具与真实命令行验证。

阶段 10（LangGraph 状态与确定性工作流）已完成并通过验收，含四类意图的真实路由验证与「模型无法跳过确认节点」的结构性证明。

阶段 11（多轮记忆、Checkpoint 与人工确认）已完成并通过验收，含真实模型的跨进程续聊、跨用户隔离与取消不写库验证。

阶段 12（Streamlit 产品界面）已完成并通过验收，含真实服务进程启动、三页无异常渲染与界面内的确认门禁验证。

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

阶段 9 证据（2026-09-11 实际执行）：

- 代码提交：`4f366dd`。
- `ruff check`：All checks passed；`mypy`（strict，files = ["src"]）：Success: no issues found in 43 source files。
- `pytest`：307 passed, 2 skipped（在 DSH 沙箱内执行）；另有 19 个使用 `tmp_path` 的用例受沙箱限制，由学习者在本地确认，两处合计 326 passed, 2 skipped。
- 模拟数据：`data/business/seed.json`（3 个用户、5 台设备、4 张订单，电话字段已掩码），随代码提交；文件中带有「模拟数据」声明。保修到期日由「激活日期 + 整数月」计算，需要判定时必须传入 `as_of` 日期，不使用随机值。
- 真实命令行验证（`rag-agent tool`，离线执行）：

  | 场景 | 结果 |
  |---|---|
  | `tool list` | 退出码 0，列出 4 个工具及其参数 Schema |
  | 查询本人新设备 D2002 | 退出码 0，`warranty_active` 为 true，到期日 2028-01-10 |
  | 查询本人旧设备 D2001 | 退出码 0，`warranty_active` 为 false（到期日 2026-03-15） |
  | 查询他人设备 D2003 | 退出码 1，`permission_denied`，`retryable` 为 false |
  | 创建工单但未确认 | 退出码 1，`confirmation_required`，数据库中没有写入 |
  | 创建工单且已确认 | 退出码 0，`created` 为 true，工单号 `T15E4459A` |
  | 重复同一请求 | 退出码 0，`created` 为 false，返回**同一个工单号** |
  | 为他人设备创建工单 | 退出码 1，`permission_denied`，未写入 |

- 契约要点：工具失败一律返回 `status="error"` 加稳定 `code` 与 `retryable`，**不会把错误伪装成正常字符串**；参数先经 Pydantic 校验再触达仓储；写工具在 `confirmed` 为 false 时拒绝执行，因此模型无法自行创建工单；幂等键可由调用方提供，否则由「用户 + 设备 + 故障 + 联系方式」派生，重复请求复用同一张工单。
- 边界说明：本阶段的 `confirmed` 字段只是工具层契约，**真正的用户确认节点属于阶段 11 的 LangGraph Interrupt**；权限模型也只做“调用方必须是设备属主”一条，不含角色与权限表。

阶段 10 证据（2026-09-11 实际执行）：

- 代码提交：`c685b7e`。
- `ruff check`：All checks passed；`mypy`（strict，files = ["src"]）：Success: no issues found in 48 source files。
- `pytest`：342 passed, 2 skipped（在 DSH 沙箱内执行）；另有 19 个使用 `tmp_path` 的用例受沙箱限制，由学习者在本地确认，两处合计 361 passed, 2 skipped。
- 真实图执行验证（`rag-agent agent`，真实模型）：

  | 场景 | 退出码 | 结果 | 节点轨迹 |
  |---|---|---|---|
  | 知识问答「制造商地址在哪里」 | 0 | `answered`，intent 由模型判定为 knowledge，置信度 0.95，引用 1 条 | `classify:knowledge` → `knowledge:answered:citations=1` |
  | 设备查询「我的机器还在保修吗」 | 0 | `answered`，intent 为 device，调用 `device.lookup` 与 `order.lookup` | `classify:device` → `device:ok` |
  | 工单请求（未确认） | 1 | `pending_confirmation`，`confirmed` 为 false，`created_ticket` 为 null，**数据表工单数不变** | `classify:ticket` → `ticket_collect:complete` → `ticket_pending:awaiting_confirmation` |
  | 同一请求加 `--confirm` | 0 | `ticket_created`，工单号 `T9AEBF945`，工单数由 1 增至 2 | 在上述轨迹后追加 `ticket_create:created=True` |

- **验收条件的结构性证明**：测试直接读取编译后图的边集合，断言**不存在** `ticket_collect → ticket_create`、`classify → ticket_create`、`clarify → ticket_create` 三条边，唯一进入创建节点的边来自待确认节点。行为测试与此互补：未确认的运行中轨迹**根本没有 `ticket_create`**。
- 两道独立屏障：图上的边限制，以及工具契约里的 `confirmed` 校验。任一道都能单独阻止未确认写入，测试分别覆盖。
- 确定性与模型边界：模型只做两件事——判断意图、从用户消息中读取字段；路由、业务顺序、澄清上限与写操作门禁全部由图和代码决定。意图输出非法或置信度低于 0.5 时一律降级为 `unknown` 并走澄清分支。
- 循环限制：状态里带澄清轮次计数（默认 3 轮，超出返回 `clarification_limit` 并转人工），图调用同时设置 `recursion_limit=12`。**本阶段图中刻意没有环**：没有 Checkpointer 时单次运行收不到新的用户输入，因此澄清分支是「返回给调用方」而不是「原地重试」；阶段 11 会用 Interrupt 把它换成真正的暂停与恢复。

阶段 11 证据（2026-09-11 实际执行）：

- 代码提交：`3f9fe05`（22 个文件，+2500 / −97）、`f5238f3`（URI 目标解析修复）、`5799bff`（设备号确定性抽取）、`95ef056`（追问改判规则）；提交时受控文件 123 个。
- `ruff check`：All checks passed；`ruff format --check`：113 files already formatted。
- `mypy`（strict，files = ["src"]）：Success: no issues found in 54 source files。
- `pytest`：**423 passed, 2 skipped**（在 DSH 沙箱内一次性跑完全部用例，0 failed）。
- 真实模型端到端验收（离线脚本连续调用 CLI，真实 32 页说明书 34 个分片的索引）：

  | 场景 | 退出码 | 关键结果 |
  |---|---|---|
  | 第 1 轮「制造商地址在哪里」 | 0 | `answered`，引用第 32 页，`checkpoints_before=0 → after=4` |
  | **第 2 轮「那它的售后电话是多少」（独立进程）** | 0 | `answered`，售后电话 400-886-8888，引用第 27 页，**`checkpoints_before=5`**，说明第二个进程读到了第一个进程写入的检查点；`window_size=2` |
  | 同一用户另一 thread | 0 | 正常回答，`window_size=0`，证明线程之间不共享上下文 |
  | U1002 使用 U1001 的 thread | 2 | `thread_ownership_conflict`，未产生任何回答 |
  | 工单请求（未确认） | 1 | `pending_confirmation`，`paused=true`，轨迹止于 `ticket_pending:awaiting_confirmation`；工单数 2 → 2 |
  | 同线程 `--cancel` | 0 | `ticket_cancelled`，轨迹追加 `ticket_create:cancelled`；**工单数仍为 2** |
  | 新线程 `--confirm` | 0 | `ticket_created`，工单号 `T11BEE6D1`；工单数 2 → 3 |
  | `thread list --user-id U1001` | 0 | 列出 4 条会话，含消息数与检查点数 |
  | `thread clear --thread-id T-acc-11-B` | 0 | `checkpoints_removed=5` |

- **验收条件逐条对应**：「服务重启后可以继续指定会话」由第 2 行（另一个进程、`checkpoints_before=5`）证明；「不同用户不会共享上下文」由跨用户拒绝与同用户不同线程两行共同证明；「取消确认不会执行写操作」由「未确认」与「取消」两行中工单数保持 2 证明。
- 确认顺序证据：轨迹中 `ticket_create` 只在恢复后才出现，且未确认时数据库中工单数不变——`interrupt()` 位于创建节点第一条语句，暂停早于任何工具调用。
- 结构性证明仍然成立：`ticket_pending → ticket_create` 是唯一进入创建节点的边，阶段 10 的边集合断言原样通过。
- 本次未新增运行时依赖：`langgraph 1.2.11` 与 `langgraph-checkpoint 4.2.0` 为既有依赖，Checkpointer 由本项目用标准库 `sqlite3` 实现。

**实现中发现并已修复的三个真问题**（都来自实际运行，而不是代码审查）：

1. **LangGraph 在自有线程池里写检查点**：`graph.invoke` 会从多个线程并发调用 `put`/`put_writes`，`sqlite3` 默认的线程检查直接报错；仅加 `check_same_thread=False` 又会得到 `InterfaceError: bad parameter or other API misuse`。最终用可重入锁把事务串行化并显式管理提交。
2. **隐式事务与 `BEGIN IMMEDIATE` 相遇**：Python 的 `sqlite3` 会在一条 `SELECT` 后打开隐式事务，导致「cannot start a transaction within a transaction」，因此写事务前先提交已打开的事务。
3. **待确认节点后的条件边会把暂停点吃掉**：若在边上再判断一次 `confirmation`，未确认时会走 END，`ticket_create` 根本不被执行，`interrupt()` 也就永远不会触发。改为无条件边后，确认与否完全由节点内的暂停决定。
4. **两个类对「数据库目标」的解析不一致**（提交 `f5238f3` 修复）：`SQLiteCheckpointer` 用 `uri=True` 连接，而 `ConversationStore` 没有，于是 `file:name?mode=memory&cache=shared` 这样的目标被后者当成磁盘上的字面文件名，两者静默使用了不同的数据库——不会报错，只是登记表看起来永远是空的。这是跑新测试时由项目根目录反复出现的空文件 `file` 暴露出来的，现已统一为 `uri=True`，并补了一条跨类回归测试。
5. **设备查询只认命令行参数，不认消息文本**（已修复）：`device` 节点原本只读 `state["device_id"]`，该字段仅由 `--device-id` 写入，因此「D2002 还在保修吗」在消息里说出来时，节点会退化成「列出该用户全部设备」。已加入确定性抽取：调用方参数优先，其次从**当前消息**、再次从**对话窗口**里取形如 `D` + 数字的设备号；抽取结果不做归属预校验，直接交给工具，因此他人设备仍然是 `permission_denied` 而不是被静默跳过。

**已修复的追问回归（提交 `95ef056`）**：

- 现象：同一句追问「那它的保修期是多久」，**没有历史时 3/3 判为 `device` 并正常回答**；**前面有一轮设备对话后 3/3 判为 `knowledge` 并拒答**。方向稳定，不是随机波动。
- 排除的原因：不是检索问题（`search` 显示含答案的第 27 页在 top-5 内）；也不是设备号丢失（`5799bff` 的确定性抽取在两个场景下都取到了 `D2002`）。根因是 `classify` 每轮都从零判定意图，历史进入 Prompt 后模型的判定发生了改变。
- 修法（代码驱动，非再调一次模型）：在 `classify` 之后加一条四条件都成立的改判规则，命中时把意图改为 `device`，并在轨迹中留痕 `classify:device:rerouted-from-knowledge`，使改判永远不会被误认为模型自己的判断。四条件是：意图可被替换（`ticket` 永不改判）、设备号已可从消息或窗口解析、当前消息是指代式追问、当前消息不含产品知识话题词。
- 写测试时发现两处会自我抵消的配置：`保修` 与时长词（`多久`、`多长时间`）原本被列为"知识话题词"，会把这条规则要修的追问直接掐掉。保修状态与"还有多久"只能从设备存储回答，因此已从该表中移除；代价是像「那它的耗电量是多少」这类真正的知识问题仍会走 `knowledge`（宁可漏改，也不劫持）。
- 真实模型复验（4 个场景）：有历史的追问 → `device`、答出 `2028-01-10`、轨迹含改判记录；反例「那它的耗电量是多少」→ 仍为 `knowledge`、无改判；冷启动同一句 → 行为不变，且因窗口为空无法解析设备号而不触发改判；「帮我把 D2001 报修」→ 仍为 `ticket` 并停在待确认。
- 依赖证明：把规则关闭后重新跑同一场景，结果回到 `knowledge` 拒答，说明这条修复确实由该规则产生。

阶段 12 证据（2026-09-11 实际执行）：

- 代码提交：`e2ca5e9`（阶段 12 功能与测试）。
- 依赖：新增 `streamlit 1.63.0`（`uv add streamlit`），其依赖 pandas 3.0.5、pyarrow 25.0.1 在 Python 3.13 下均有官方 cp313 wheel。
- `ruff check`：All checks passed；`ruff format --check`：122 files already formatted。
- `mypy`（strict，files = ["src"]）：Success: no issues found in 61 source files。
- `pytest`：**440 passed, 2 skipped**（在 DSH 沙箱内一次性跑完全部用例，0 failed）。
- 真实服务启动：`python -m streamlit run src/rag_agent/ui/app.py --server.port=8555`，`GET /` 返回 **HTTP 200**，`GET /_stcore/health` 返回 **200 ok**。
- 三页渲染验证：用 Streamlit 官方 `st.testing.v1.AppTest` 逐页执行（入口 + `switch_page` 到三个页面），四者 `at.exception` 均为空。
- 界面内行为验证（注入假模型，不联网络）：

  | 场景 | 结果 |
  |---|---|
  | 对话页提问「D2002 还在保修吗」 | 调用 `device.lookup`，答出「2028-01-10」 |
  | 报修请求 | 出现待确认，`pending_action.action` 为 `ticket.create`，工单数仍为 0 |
  | 点击「取消」 | `pending_action` 清空，**工单数仍为 0** |
  | 点击「确认创建」 | 工单数变为 1 |
  | 知识库页上传 Markdown | 文件写入 `data/raw/`，元数据中出现该来源 |
  | 检索页查询已有分片 | 显示最高相似度、前两名分差与判定依据 |
  | 打开流式开关提问 | 走知识路径（`intent_source=stream`），且不写检查点 |

**实现中发现并已修复的三个真问题**：

1. **`MetadataStore` 与 `BusinessRepository` 同样缺少跨线程许可**：`AppTest` 每个页面在不同线程执行，立刻复现 `SQLite objects created in a thread can only be used in that same thread`。两者现与 Checkpointer 一致：`check_same_thread=False` + 可重入锁串行化事务，文件库启用 WAL，使同库的另外两个连接可以在写入期间读取。
2. **检索页在只有一个命中时崩溃**：此时 `margin` 为 `None`，而页面用 `f"{margin:.4f}"` 格式化，抛 `TypeError`。现显示为「不适用（只有一个命中）」。
3. **侧边栏把会话编号同时用作 `session_state` 键与 `text_input` 的 key**：点击「新建会话」时赋值触发 `StreamlitWidgetAlreadyInstantiatedError`。现改为输入框使用独立键，并通过 `on_change` 回调同步，符合 Streamlit 对「渲染后不得改写控件键」的要求。

已知环境注意事项：

- 在 DSH 沙箱内运行 pytest 时，pytest 创建目录用的 `tempfile.mkdtemp()`（`.venv/Lib/site-packages/_pytest/cacheprovider.py:66`）和 `mkdir(mode=0o700)`（`.venv/Lib/site-packages/_pytest/pathlib.py:232`）都会产生沙箱进程之后无法访问的目录，表现为 `PytestCacheWarning` 或使用 `tmp_path` 的用例报 `PermissionError`。**阶段 11 已用 `tests/conftest.py` 覆盖内置 `tmp_path`，把每个用例的临时目录放到项目内的 `.pytest_tmp/`（已被 .gitignore 忽略）**，因此沙箱内不再出现该错误；`.pytest_tmp/` 与 `pytest-cache-files-*` 仍属沙箱副作用，可安全删除。
- 沙箱内 uv 无法写入用户级缓存（`AppData\Local\uv\cache`），需设置 `UV_CACHE_DIR` 指向项目内目录；该目录已在 `.gitignore` 中忽略。
- 用 PowerShell 的 `Get-Content` / `Set-Content` 改写 `.py` 文件会破坏非 ASCII 字符（本项目已实际踩到一次：em dash 变成乱码）。改动源码请使用编辑工具，而不是 PowerShell 文本往返。

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
- 可预测的模拟业务数据（用户、设备、订单），带掩码联系方式与按整数月计算的保修状态。
- 四个契约化业务工具：用户查询、设备查询、订单查询与工单创建，失败一律返回结构化错误码而不是字符串。
- 工具层强制确认契约：未确认的写请求被拒绝，数据库不产生任何写入。
- 工单创建幂等：重复请求返回同一张工单，并明确区分 `created` 为 true 或 false。
- 单条权限规则：调用方必须是设备属主，越权返回 `permission_denied`。
- `rag-agent tool list` 与 `rag-agent tool <名称> --args '{...}'`，可让每个工具脱离 Agent 独立调用与验证。
- LangGraph 工作流：意图路由、知识问答、设备查询、工单收集四条确定性路径，节点轨迹可回放。
- 模型边界清晰：模型只判断意图与读取字段，非法或低置信度输出一律降级为 `unknown` 并转澄清。
- 写操作门禁：图上的边不允许从收集节点直达创建节点，工具层再次校验确认标记。
- 澄清轮次上限与 `recursion_limit` 双守卫，超出即转人工而不是无限追问。
- `rag-agent agent "问题" --user-id U1001` 单轮跑图，输出状态、意图、引用、工具结果、待确认动作与节点轨迹。该命令无状态，工单请求停在待确认状态且不写库。
- 自实现 SQLite Checkpointer：继承 LangGraph 的 `BaseCheckpointSaver`，按通道版本保存状态，可跨进程恢复，且不引入额外运行时依赖。
- `rag-agent chat "问题" --thread-id T1 --user-id U1001` 多轮会话：按 `thread_id` 隔离，最近 8 条消息进 Prompt 而完整历史留在 Checkpoint 中。
- 服务重启后继续同一会话：两个独立进程读写同一个检查点库，第二个进程能读到第一个进程写入的状态。
- 人工确认为图上真实暂停：`ticket_create` 节点第一条语句是 `interrupt()`，暂停早于任何工具调用；`--confirm` 从断点恢复并写库，`--cancel` 恢复后不产生任何写入。
- 会话归属校验：跨用户使用同一 `thread_id` 返回 `thread_ownership_conflict` 且退出码 2。
- 短期记忆与长期记忆分离：消息窗口只影响 Prompt，长期偏好按 `user_id` 存在 `user_preferences` 表，清除会话不会删除偏好。
- `rag-agent thread list|clear|preferences`：列出会话（含消息数与检查点数）、清除指定会话并返回删除数量、查看或写入长期偏好。
- 会话消息在每轮结束后通过 `update_state` 追加，使下一轮能读取上一轮的用户消息与助手回复。
- Streamlit 三页界面：对话、知识库管理、检索调试，用 `st.navigation` 组织，可通过 `uv run streamlit run src/rag_agent/ui/app.py` 启动。
- 对话页支持流式生成（带会话记忆与引用校验）、引用卡片、工具调用步骤，以及待确认写操作的确认/取消按钮。
- 界面明确区分「本次浏览器会话显示的消息」与「Checkpoint 中持久化的会话状态」，并在侧边栏同时给出索引分片数与文档数。
- 界面复用 CLI 的同一批入口（`chat_turn`/`stream_chat_turn`/`ingest_path`/`sync_index`/`remove_document`），因此不能绕过写操作门禁。
- 知识库页支持上传、查看、删除与重建索引，且每次 rerun 重新读取真实状态，提交结果用一次性会话值传递以避免重复入库。
- 流式路径与一次性路径构造同一条提示词（最近对话 + 长期偏好 + 编号资料），由测试固定。

## 尚未实现

- 安全与可观测性降级（阶段 13）、Docker 与最终交付（阶段 14）。
- 界面身份认证：侧边栏的「用户编号」只是演示标识，任何访问者都能填任意编号。
- 由用户文本「确认/取消」触发恢复：当前必须用 `--confirm` / `--cancel` 或界面按钮显式表达，以避免让模型决定是否写库。
- 矢量轮廓 PDF 的文本提取（需要 OCR，当前明确不支持）。
- 重排：已确认阈值与分差都不可分，是否需要重排需靠 Recall@K 与 MRR 的进一步实验判断。
- 长分片的主题稀释问题（第 2 页 750 字符分片导致两条安全类问题漏检），需要按编号条目二次切分的实验。
- 表格序号与上下文编号混淆导致的引用越界（6 条），需要更强的格式约束或校验提示。
- 角色与权限表：当前只有「属主」一条规则，模拟定位下够用但不足以表达更细的授权。
- 界面自动化覆盖的边界：`AppTest` 无法触发图表/表格的**选择**事件，也无法验证自定义组件 JavaScript 与最终视觉效果，这部分仍需人工查看页面。

## 最近结构变化

本次同步（阶段 12）相对上一次的主要变化：

- 新增 `src/rag_agent/ui/`：`app.py` 入口与路由、`services.py` 共享资源、`app_pages/` 三个页面脚本。
- `pyproject.toml` 新增 `streamlit>=1.57,<2`；`uv.lock` 同步（streamlit 1.63.0）。
- `src/rag_agent/agent/graph.py`：新增 `stream_chat_turn` 流式入口（带会话记忆与引用校验），`ChatTurn` 新增 `turn_trace` 与 `turn_tool_results` 区分本轮回合与整条会话。
- `src/rag_agent/generation/rag_answer.py`：`stream_raw_answer` 显式接收 `conversation_context`，流式路径不再丢失记忆。
- `src/rag_agent/storage/sqlite.py`、`src/rag_agent/storage/business.py`：连接改为 `check_same_thread=False` + 可重入锁，文件库启用 WAL。
- `.gitignore` 新增 `.uv-cache/`（沙箱内 uv 缓存落在项目内时忽略）。
- `tests/unit/` 新增 `test_ui_app.py` 与 `test_rag_answer_stream_context.py`。

上一次同步（阶段 11）的主要变化：

- 新增 `src/rag_agent/memory/`：`schema.py` 保存建表语句，`checkpoints.py` 是自实现的 SQLite Checkpointer，`conversation.py` 负责会话归属、消息窗口与长期偏好。
- `src/rag_agent/agent/nodes.py`：创建工单节点以 `interrupt()` 作为第一条语句，暂停早于任何工具调用；字段抽取改读整个 state 以使用最近对话。
- `src/rag_agent/agent/graph.py`：可挂载 Checkpointer，新增 `ChatTurn` 与 `chat_turn()` 支持按线程运行与 `Command(resume=...)` 恢复；**待确认节点之后由条件边改为无条件边**，确认与否交给节点内的暂停决定。
- `src/rag_agent/agent/intent.py`：`classify_intent` 接受最近对话作为 Prompt 上下文（路由仍只看返回标签与置信度）。
- `src/rag_agent/generation/rag_answer.py`：用户消息可前置「最近对话 + 长期偏好」区块，并明确标注该区块不作为事实依据。
- `src/rag_agent/config/settings.py`：新增 `conversation_window_size` 与 `checkpoint_path`。
- `src/rag_agent/__main__.py`：新增 `chat` 与 `thread` 命令及 `--thread-id`、`--cancel`、`--preference`、`--window`；`agent` 命令移除 `--confirm`（恢复暂停属于 `chat`）。
- `tests/conftest.py`：覆盖内置 `tmp_path`，把临时目录放到项目内 `.pytest_tmp/`，使沙箱内也能正常跑完全部用例。
- `tests/unit/` 新增四个测试文件与一个测试辅助模块（`memory_test_support.py`）。
- `docs/adr/0003-conversation-memory.md`：记录 Checkpointer 实现方式、Interrupt 位置与记忆边界的决策。

本次同步的第二轮（同一阶段的修复提交，`f5238f3`、`5799bff`、`95ef056`）：

- 新增 `src/rag_agent/agent/devices.py`（设备号解析的唯一来源）与 `src/rag_agent/agent/routing.py`（模型判定之后的确定性改判规则）。
- 新增 `tests/unit/test_agent_routing.py`，`test_agent_memory.py` 增加设备解析与改判留痕用例。
- `src/rag_agent/agent/nodes.py`：`classify` 应用改判并在轨迹留痕 `classify:device:rerouted-from-*`；`device` 节点改用共享的设备号解析。
- `src/rag_agent/agent/__init__.py`：对外暴露 `prefer_device` 与 `resolve_device_id`。

上一次同步（阶段 10）的主要变化：

- 新增 `src/rag_agent/agent/`，承载状态、意图识别、节点与图。
- `src/rag_agent/generation/rag_answer.py` 抽出公共的 `extract_json_object`，供意图识别复用。
- `src/rag_agent/__main__.py` 新增 `agent` 命令与 `--user-id`、`--device-id`、`--contact`、`--confirm`。
- `tests/unit/` 新增四个测试文件与一个 Agent 测试辅助模块。
