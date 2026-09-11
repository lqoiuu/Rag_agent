# 企业售后知识库 RAG 智能体

这是一个面向扫地机器人售后场景的学习与作品集项目。知识资料可以来自有权使用的说明书、维修手册和 FAQ；用户、设备、订单及工单接口均为本地模拟实现，不代表真实企业系统。

当前状态：阶段 14（引用修复、端到端验收与 Docker 交付）已完成并通过验收。

## 当前技术基线

- Python 3.13
- uv 与 pyproject.toml 管理项目和依赖
- LangChain、LangGraph、Chroma
- Streamlit 提供界面（只负责展示与触发，业务规则仍在 `agent/`、`tools/`、`memory/`、`ingestion/`）
- 通义千问，通过 DashScope 的 OpenAI 兼容端点经 httpx 调用
- pypdf 解析 PDF 文本，fonttools 补齐 CFF 字体编码解析
- langchain-text-splitters 负责递归字符切分
- pydantic-settings
- 会话持久化使用 langgraph-checkpoint 的 `BaseCheckpointSaver` 与标准库 `sqlite3` 自实现，不额外引入 checkpoint 后端依赖
- pytest、Ruff、mypy
- Docker Compose 提供可复现的非 root 容器交付

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

## 启动与关闭

以下命令都在 `rag-agent-assistant/` 项目根目录执行。Docker 方式用于完整交付验收，本地方式适合开发调试；两种方式不要同时占用 8501 端口。

### Docker Compose（推荐）

前置条件是 Docker Desktop 的 Linux Engine 已启动，且项目根目录存在本地 `.env`。Compose 只把配置注入运行中的容器，不会把 `.env` 或真实密钥复制进镜像。只使用 `docker compose config --quiet` 做校验；不带 `--quiet` 会把解析后的环境变量（包括密钥）打印到终端。

首次启动或代码发生变化时执行：

```powershell
docker compose config --quiet
docker compose up --detach --build --wait
docker compose ps
docker compose exec --no-TTY rag-agent rag-agent health
```

成功时 `docker compose ps` 显示容器为 `healthy`，应用健康输出中的 `status` 为 `ok`。浏览器访问 `http://127.0.0.1:8501`；端口只绑定到本机，不对局域网公开。当前已验证镜像构建、服务启动、容器健康检查与三页界面渲染。容器以 UID/GID 10001 运行，应用根目录只读，只有三个数据卷和临时 `/tmp` 可写。

关闭并移除容器及项目网络：

```powershell
docker compose down
```

该命令不会删除保存原始资料、Chroma 索引和 SQLite 状态的三个命名卷。除非明确要清空全部容器数据，否则不要增加 `--volumes` 参数。

若受限网络导致 BuildKit 首次获取 Docker Hub 令牌失败，应先在 Docker Desktop 中配置可用代理，再预拉取基础镜像后重建：

```powershell
docker pull python:3.13-slim
docker compose build
```

### 本地虚拟环境

完成前面的 `uv sync` 并配置本地 `.env` 后启动：

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/rag_agent/ui/app.py
```

浏览器打开终端输出的地址（默认 `http://localhost:8501`）。关闭时回到运行该命令的终端按 `Ctrl+C`；这只停止 Streamlit 进程，不会删除 `data/` 下的原始资料、索引或 SQLite 状态。

界面分三页：

- **对话**：流式生成、引用来源、工具调用步骤，以及待确认写操作的「确认 / 取消」按钮。
- **知识库**：上传（TXT / Markdown / PDF）后立即入库，可查看已入库文档、删除、重建索引。
- **检索调试**：查询命中的分片、相似度、页码、前两名分差与置信判定依据。

界面里的「用户编号」只是演示标识，**不是身份认证**；真实登录与授权尚未实现。会话状态保存在 SQLite Checkpoint 中，刷新页面不会丢会话，但界面显示的消息列表会清空。

## 目录职责

- src/rag_agent/：应用源码。
- tests/：自动化测试。
- docs/：稳定的需求、架构、术语和架构决策。
- data/：本地运行数据。`raw/` 存放原始资料，`chroma/` 与 `rag_agent.sqlite3` 保存向量索引与元数据，全部不提交到 Git。
- pyproject.toml：项目元数据、直接依赖和工具配置。
- uv.lock：完整且精确的依赖版本。
- .env.example：环境变量示例，不包含真实密钥。
- Dockerfile：Python 3.13、锁定依赖、非 root 用户和 Streamlit 启动入口。
- compose.yaml：端口、健康检查、环境变量与三个持久卷。
- .dockerignore：排除密钥、虚拟环境、测试报告和本地运行数据。

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
- 模拟业务数据与四个契约化工具：用户查询、设备查询（含保修状态）、订单查询与工单创建。
- 工具失败返回稳定错误码与可重试标记，不把错误伪装成正常结果；写工具在未确认时拒绝写入，重复请求返回同一张工单。
- `rag-agent tool list` 与 `rag-agent tool <名称> --args '{...}'`，可脱离 Agent 单独调用任意工具。
- `rag-agent agent "问题"` LangGraph 工作流：意图路由到知识问答、设备查询或工单流程，输出状态、引用、工具结果与节点轨迹。该命令**无状态**，工单请求停在待确认状态且不写库。
- `rag-agent chat "问题" --thread-id T1 --user-id U1001` 多轮会话：按 `thread_id` 隔离，会话状态由 SQLite Checkpointer 持久化，**进程重启后可以继续同一个会话**；最近 8 条消息进 Prompt，完整历史留在 Checkpoint 中。
- 工单确认是图上的真实暂停：`ticket_create` 节点第一件事就是 `interrupt()`，暂停发生在任何工具调用之前。`rag-agent chat --thread-id T1 --user-id U1001 --confirm` 从断点恢复并创建工单；`--cancel` 恢复后不执行任何写入，两者互斥。
- `rag-agent thread list|clear|preferences --user-id U1001`：查看会话、清除某个会话（返回删除的 Checkpoint 数量）、查看或写入长期偏好。
- 长期偏好按 `user_id` 保存，清除会话不会删除偏好；跨用户使用他人的 `thread_id` 会被拒绝（`thread_ownership_conflict`，退出码 2）。
- 写操作有两道独立屏障：**图上的边不允许从信息收集直达创建节点**，且 `ticket_create` 节点先暂停等人确认；工具层再校验一次确认标记，任一道都能单独挡住未确认写入。
- Streamlit 三页界面（对话 / 知识库 / 检索调试），复用与 CLI 相同的入口，因此不能绕过写操作门禁。
- 对话页区分**本轮回合**与**整条会话累积**的轨迹和工具结果：Checkpoint 里存的是累积值，界面另算本轮的增量，避免把历史当成刚刚发生的事。
- 知识库页在每次 rerun 重新读 SQLite 而不是缓存列表；上传提交的结果用一次性会话值传递，防止 rerun 造成重复入库。
- 检索调试页把「阈值不能区分该答与不该答」这一实测结论直接写在页面上，避免读者把分数当成正确性判据。
- 信任边界：被索引文档的正文显式包裹为不可信数据，系统提示声明块内不是指令；形似指令的文字会被上报但不改变回答。**真正的防线是引用校验**——文档无法让系统产出可核验的假引用。
- 工具权限白名单与角色：`end_user` 只能读自己的记录，`support_agent` 可跨用户读取；角色由调用方声明（CLI `--role`、界面选择器），只能下调，未知值落回最严格角色。
- 运行期指标：本轮总耗时、检索命中数与最高分、工具调用与失败码、注入疑点，`rag-agent chat` 输出与界面指标面板都能看到。
- 统一降级策略：模型、工具与检索失败映射为 retry / answer_partial / refuse / report_error 四种动作与面向用户的措辞，失败绝不伪装成回答。
- 引用二次校验与修复：模型只选择 `SOURCE-A` 等临时来源标签，系统再映射回已检索且已校验的引用编号；不会重写答案事实，也不会修复混入虚构来源的回答。
- Docker Compose 交付：Python 3.13 镜像以 UID 10001 非 root 用户运行，原始资料、Chroma 与 SQLite 分卷持久化，并提供 Streamlit 健康检查。

## 模拟数据边界

`data/business/seed.json` 中的用户、设备、订单，以及所有工具返回值，**均为模拟数据**，不代表任何真实个人或企业信息；联系方式已做掩码处理。README、演示界面与简历中都应保留这一声明。

## 当前评测基线

| 指标 | 数值 | 说明 |
|---|---|---|
| Recall@5 | 0.9535 | 43 条可回答问题命中 41 条（按页粒度） |
| MRR | 0.8205 | 正确分片平均排在较前位置 |
| 回答率 | 0.9302 | 阶段 14 两次 49 条 Qwen 评测结果相同；阶段 13 基线为 0.8140 |
| 引用正确率 | 0.8250 | 第二次阶段 14 评测；第一次为 0.7750，阶段 13 基线为 0.7429 |
| 拒答正确率 | 1.0 | 6 条不可回答问题全部正确拒答 |
| 决策正确率 | 0.9388 | 阶段 14 两次结果相同；阶段 13 基线为 0.8367 |
| 引用修复 | 6 / 6 | 第二次评测全部修复成功；第一次为 5 / 5 |
| 忠实度（代理） | 0.5769 | 第二次评测结果，答案为引用分片全文覆盖的比例，属弱指标 |
| 回答耗时 P50 | 1892.8 ms | 第二次阶段 14 的 49 条逐条实测耗时，线性插值分位数 |
| 回答耗时 P95 | 3910.2 ms | 同上；包含首次回答及按条件触发的引用校对耗时 |

阶段 13 基线来自 `answer-20260911T080858Z.json`，阶段 14 两次对照来自 `answer-20260911T090138Z.json` 与 `answer-20260911T090839Z.json`。检索指标确定可重复，回答指标存在模型运行波动；第二次引用正确率高于第一次，部分来自 q21、q22 的模型输出变化，不能全部归因于修复代码。q34 修复选择的第 25 页与第 8 页分别支撑答案中的不同事实，但评测集只允许第 25 页，因此严格口径仍判为引用错误。

## 尚未实现

- 界面身份认证与更细的写权限：角色与用户编号都是自述，任何访问者都能填。
- token 用量未并入指标：它记录在日志里，尚未写回图状态。
- 矢量轮廓（文字转曲线）PDF 的文本提取，需要 OCR，当前明确不支持。

**已知检索限制**：49 条实测数据显示，可回答与不可回答问题的最高相似度区间重叠 0.1082，前两名分差同样重叠，因此 `RAG_AGENT_RETRIEVAL_THRESHOLD` **不存在能把两组分开的阈值**，它只用于避免无意义的模型调用；拒答由「只能依据资料回答」的约束与引用校验承担。目录类分片已用确定性规则在入库时过滤（前两名分差由 0.0011 提升到 0.0375）。另有两条安全类问题因第 2 页分片过长、主题被稀释而漏检，属已知待改进项。

模型联网调用需要本地 `.env` 中的 `RAG_AGENT_QWEN_API_KEY`；相关集成测试默认跳过，只有设置 `RAG_AGENT_RUN_LIVE_TESTS=1` 时才访问网络。

具体操作步骤由 Codex 在对话中逐步给出，不在 README 中维护阶段执行流程。
