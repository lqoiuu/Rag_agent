# ADR-0003：会话持久化用自实现 SQLite Checkpointer，人工确认用节点内 Interrupt

- 状态：已接受
- 决策日期：2026-09-11
- 适用阶段：阶段 11 起
- 决策范围：多轮记忆的存储方式、人工确认的暂停机制、短期记忆的边界
- 关联决策：ADR-0001 技术栈、ADR-0002 模型接入层

## 背景

阶段 11 的验收条件是三条硬要求：服务重启后能继续指定会话；不同用户不共享上下文；取消确认不执行写操作。

阶段 10 已经实现 LangGraph 工作流，但那份图刻意没有环：没有 Checkpointer 时，单次运行收不到新的用户输入，所以澄清分支只能「返回给调用方」；工单确认也只是一次性的 `pending_confirmation` 状态，下一轮无法真正从断点继续。因此本阶段需要解决三件事：

1. 会话状态放在哪里，以及它如何跨进程存活。
2. 人工确认用什么机制暂停与恢复，且不能削弱阶段 10 已有的结构性证明。
3. 多轮上下文进 Prompt 的边界在哪，长期信息放哪。

主要约束：

- 阶段 10 的验收测试直接断言**编译后图的边集合**中不存在「收集 → 创建」的直达边。新机制不能把这条证明变模糊。
- 写操作必须在确认之前不发生，包括「确认被拒绝」和「恢复后失败」两种情况。
- 依赖越少越好，且必须在 Python 3.13 下可用。
- 会话必须真的能在进程结束后恢复，所以内存实现不足以作为交付方案。

## 决策

1. **自实现 SQLite Checkpointer**：继承 `langgraph.checkpoint.base.BaseCheckpointSaver`，用标准库 `sqlite3` 实现 `get_tuple`、`list`、`put`、`put_writes`、`delete_thread`，序列化复用 LangGraph 的 `JsonPlusSerializer`。不引入 `langgraph-checkpoint-sqlite`。
2. **三张表按 LangGraph 的语义拆分**：`checkpoints` 存每个 super-step 与父指针，`checkpoint_blobs` 按 `(thread, namespace, channel, version)` 存通道值，`checkpoint_writes` 存按任务写入（含保留的 `__interrupt__` 通道）。表结构放在 `memory/schema.py`，因为检查点与对话登记表都要建表。
3. **人工确认放在节点内 `interrupt()`**，且是 `ticket_create` 的第一条语句；恢复时由调用方传入 `Command(resume={"confirmed": true|false})`。图的边集合因此完全不变，`ticket_pending → ticket_create` 仍是唯一进入创建节点的边。
4. **`ticket_pending` 之后改成无条件边**，删掉原先那个「未确认就直接结束」的条件边。确认与否是节点的职责，不是边的职责；图上的暂停点因此可见，而不是被一条静默的 END 分支取代。
5. **短期记忆窗口默认 8 条**：进 Prompt 的只有最近 8 条消息，完整历史保存在 Checkpoint 中。**长期偏好另存 `user_preferences` 表**，按 `user_id` 归属，清除会话不会删除偏好。
6. **会话归属由应用层维护**：`_threads` 表记录 `thread_id → user_id`。LangGraph 把 `thread_id` 当作不透明字符串，所以「这条会话属于谁」必须由知道用户概念的代码来判定，跨用户使用同一 `thread_id` 直接拒绝。

## 决策理由

### 为什么不装 `langgraph-checkpoint-sqlite`

官方 SQLite 后端会再拉入 `aiosqlite` 与 `sqlite-vec`（含 C 扩展），而本项目阶段 11 要讲清楚的知识恰恰是「Checkpoint 到底存了什么、为什么这样恢复」。自己实现一遍的代码量约 300 行，且能复用项目里已有的两处标准库 `sqlite3` 实践（`storage/sqlite.py`、`storage/business.py`），风格统一。代价是一套自测，这部分由 `test_memory_checkpoints.py` 承担。若未来需要异步或多进程写入，再换成官方后端只是替换一个类。

### 为什么不用 `interrupt_before=["ticket_create"]`

`interrupt_before` 只在「未编译的图」上生效；编译之后边集合里 `ticket_pending → ticket_create` 依然存在，阶段 10 那条「不存在直达创建的边」的断言会失去意义——它断言的仍是同一组边，但真正拦住写入的东西变成了编译参数而不是图结构。节点内 `interrupt()` 不碰边集合，阶段 10 的测试原样通过，暂停点还写在了节点里，可读性更好。

另一个好处是恢复值的投递方式：`interrupt()` 的返回值由 LangGraph 直接交给暂停的那个节点，调用方不需要为「用户确认结果」人为构造一个 state 字段再塞进去。

### 为什么 `interrupt()` 必须是节点的第一条语句

`interrupt()` 在首次执行时抛出来暂停运行；恢复时节点会**从头重新执行**。所以暂停必须排在所有副作用之前，否则恢复时侧效应会执行两次。把 `interrupt()` 放在创建工具调用之前，取消确认时节点直接返回、数据库无写入；恢复后即使后续失败，也不会留下半张工单。

节点还额外保留了 `_is_confirmed` 白名单校验：只有恢复载荷里显式 `confirmed is True` 才写库，字符串 `"true"`、缺字段、非字典一律按未确认处理。

### 为什么窗口是 8 条而不是全量

全量历史进 Prompt 有两个代价：token 随轮次线性增长，以及早期无关内容干扰本轮判断。窗口取 8 条是成本与指代能力的折中：足够覆盖「上一问 + 上一答 + 两轮追问」。关键在于**窗口只是 Prompt 边界，不是存储边界**——完整 `messages` 仍在 Checkpoint 里，需要时可以用 `graph.get_state` 或 `--window` 覆盖取回。测试用「窗口 3 条 / 库里 10 条」把这条边界钉住了。

长期偏好单独放一张表，是因为「这位客户用邮箱联系」不属于某一轮对话：清除会话不该把它一起删掉。测试 `test_thread_clear_removes_the_conversation_but_keeps_preferences` 覆盖了这一点。

### 为什么会话归属必须由应用层判定

Checkpoint 的键是 `thread_id`，它对 LangGraph 只是一段字符串。如果只依赖它做隔离，那么「不同用户不共享上下文」就退化成「调用方自觉使用不同的 id」，这不是隔离，只是约定。因此在同一个数据库里加了 `_threads` 表记录归属，`chat` 与 `thread clear` 都会先校验；跨用户访问返回 `thread_ownership_conflict` 且退出码 2。

### 为什么连接必须显式加锁

实测发现 `graph.invoke` 会在 LangGraph 自己的线程池里执行节点，因此 `put` 与 `put_writes` 会**从多个线程并发**打到同一个 saver 上。这带来两个必须处理的后果：一是 `sqlite3.connect` 的默认线程检查会直接报错，需要 `check_same_thread=False`；二是 Python 的 `sqlite3` 在单连接上并发使用并不安全，仅关掉检查会得到 `InterfaceError: bad parameter or other API misuse`。因此用一把可重入锁把事务串行化，并显式管理 `BEGIN IMMEDIATE` 与提交，避免「读语句留下的隐式事务」与「写事务」相遇。

## 考虑过的备选方案

### 方案 A：安装 `langgraph-checkpoint-sqlite`

代码最少、上游维护，但新增三个依赖（含带 C 扩展的 `sqlite-vec`），且会话格式由外部库决定。否掉的原因是依赖成本与学习目标的权衡：本阶段的核心知识正是检查点的结构。

### 方案 B：`InMemorySaver`

阶段 10 的测试已经在用，实现成本为零。但它是进程内字典，**无法满足「服务重启后可以继续指定会话」**这条验收条件，只能作为测试替身，不能作为交付方案。

### 方案 C：`interrupt_before` 编译参数

见上：会让阶段 10 的结构性证明失去意义。

### 方案 D：把确认结果直接写进 state 再重跑图

即阶段 10 的 `--confirm` 做法。它没有真正的暂停点，且「已确认」是调用方凭空塞进状态的，模型或调用方都可能绕过。否掉。

## 后果

正面：

- 会话跨进程存活，`rag-agent chat` 两次独立调用共享同一会话（实测 `checkpoints_before=5`，即第二个进程读到了第一个进程写的检查点）。
- 人工确认有真实的暂停点，且暂停发生在任何写入之前；取消确认不产生任何数据库写入（实测工单数 2 → 2 → 2）。
- 图的边集合未变，阶段 10 的结构性证明继续成立。
- 没有新增运行时依赖。
- 短期记忆与长期记忆的边界由代码与测试共同固定。

负面与限制：

- 自实现 saver 需要自己维护；它只覆盖本项目用到的同步接口，异步接口未实现。
- 单连接加锁意味着同一进程内不可能并发写两个会话，吞吐上限受此约束；这是本地学习与演示定位下的取舍。
- 会话归属是应用层约定，直接调用 `chat_turn` 的代码可以绕过校验，`chat` 命令才是带校验的入口。
- 用户输入的「确认/取消」文本尚未接入自然语言判定：当前由 `--confirm` / `--cancel` 显式表达，避免让模型决定是否写库。
