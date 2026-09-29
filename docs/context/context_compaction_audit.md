# ServicePilot 上下文审计

## 改造前

- `WorkflowState` 只携带当轮 `user_id`、`message`、`session_id`、路由、事件和响应；LangGraph 本身没有累计 `messages`。动态子图的 `ExecutionState` 携带 `TaskState`、`SessionState`、当前目标、工具结果与响应。
- `SessionState.turns` 在 SQLite／Redis 会话存储中逐轮追加，没有持久长度上限。`LayeredMemoryManager.assemble` 已有 800 token 预算，按近期消息选取，并检索最多 3 条 episode；但这只是读取时裁剪，不删除已存历史，也没有结构化摘要。episode 每轮追加，用户隔离，但无容量限制。
- 当前 LLM 路由器只收到当轮消息。动态 Planner 的模型输入是当前目标、当前订单观察、受限步骤模板，按需加 Failure Diary 的已验证 lesson；原始聊天历史没有直接进入 Planner。确定性业务执行也不读取摘要来决定工具参数。
- RAG 在当前请求检索与评估，证据进入当轮决策和引用；会话中没有累计 RAG chunk。Failure Diary 经检索器按需注入 Planner 或敏感步骤，不应整体放入摘要。Trace 记录事件，未作为模型输入。
- 最易膨胀的是 `SessionState.turns`、每轮生成的 episode、动态任务的历史与失败事件，以及事件存储。只有第一项直接属于本次轻量压缩范围；动态任务历史和 Trace 留在审计／状态存储，不进入摘要模型。

## 四类划分与改动边界

| 类别 | 当前处理 |
| --- | --- |
| Working | 当轮用户目标、近期消息、结构化会话／任务状态、当前计划、当前订单观察、当前政策证据和安全前置条件。 |
| Compressed | 旧对话的可确认短句和已完成步骤，滚动写入 `ContextSummary`。 |
| Retrieval | Failure Diary、历史政策、episode、Trace 与旧工具详情；只按现有检索途径调用。 |
| Drop | 重复／空白消息、过长原始输出、调试文本，以及失效计划和政策内容。 |

`order_id`、`user_id`、`operation_id`、退款金额与状态、订单与支付状态、工具执行 ID、授权、`plan_version` 和 `goal_version` 绝不能由摘要恢复。它们保持在 `SessionState`、`TaskState`、仓储或真实工具结果。摘要只作为规划提示，严格验证来源，不能覆盖结构化状态。政策证据继续由当前 RAG 提供，不把旧证据从摘要重新当作有效证据。

## 最小集成路径

在会话写入处追加阈值压缩，复用 DeepSeek 的 JSON 模式传输；成功才替换旧消息，失败保持原状。`LayeredMemoryManager` 可读取摘要，动态 Planner 仅在启用压缩且存在模型时把摘要及近期消息附在观察里。当前目标变化会清空旧摘要和旧目标消息，保留当轮用户消息，防止旧计划重新进入规划。会话的累计 `turn_count` 独立保存，以维持审计事件序号和界面轮数。现有 LangGraph、RAG、Failure Diary 和 Trace 主流程保持原样。
