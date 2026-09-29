# ServicePilot Failure Memory：代码审计

审计日期：2026-09-25。范围是当前 `src/servicepilot`、`tests` 和构建入口；以下描述现状，不把计划中的能力当作已实现。

| 检查项 | 现有实现 | 可用于失败判定的证据与缺口 |
| --- | --- | --- |
| LangGraph State | `graph_runtime.WorkflowState` 处理请求路由；开启动态规划时 `dynamic_runtime.ExecutionState` 包含 `TaskState`、`result`、`step_id`、事件和回复。 | `TaskState` 持久化目标、计划版本、步骤状态、订单观察、失败事件和 operation ID。没有结构化长期失败记录。 |
| 动态规划与重规划 | `DynamicTaskRuntime` 已有 observe → reconcile → plan → select → execute → evaluate → recover；`StateReconciler` 在目标或订单快照变化时使旧计划失效。`TaskPlanner.validate` 限定工具、顺序、订单和安全前置条件。 | 仅退款/换货/取消路由使用动态子图，且须启用动态开关。默认非动态路径不能假定有完整 `TaskState`。 |
| Trace | `TraceService`、`TraceStore` 独立 SQLite 保存 trace、span、event，记录延迟、token、工具调用和重规划次数；可按拥有者读取。 | Trace 保留期默认 14 天，可能因写入失败、截断或中断而不完整；敏感字段被清洗。Trace 本身不代表业务成功，也不能单独证明根因。长期经验必须保留最小证据引用和证据摘要。 |
| 工具结果 | `ToolResult(ok,data,error)`；`AgentEvent` 记录调用结果、错误码、重试次数；工具调用有 Trace span。 | 副作用调用结果可能未知；网络 timeout 或 500 不等于 Agent 失误。部分异常在恢复路径中已被转成安全升级。 |
| 成败判定 | 动态路径以 `GoalStateEvaluator` 判定计划步骤；`_evaluate` 用 `get_operation`/`get_ticket` 核对副作用，`_stop` 写 `completed`、`blocked`、`cancelled`、`escalated` 等状态。 | `completed` 仍不是外部用户目标完全成功的通用 oracle；`blocked`/`escalated` 可是正确安全行为。离线实验需要独立业务断言。 |
| SQLite | `SQLiteCommerceRepository` 已有 `commerce_orders`、`commerce_contexts`、`commerce_grants`、`commerce_operations`、`commerce_refunds`、`commerce_tickets`、`commerce_attempts`、`commerce_tasks`、`commerce_escalations`、`commerce_leases`。`SQLiteSessionStore` 有 `sessions` 和 `agent_events`；TraceStore 有 `obs_*` 表。 | 可另建独立 `failure_*` 表，不应把经验写入业务事务或只放在易清理的 Trace 表。 |
| Redis | `RedisSessionStore` 用 TTL 保存会话、事件和用户会话索引，失败时可回退 SQLite。 | 仅适合短期状态，不适合长期经验的唯一存储。 |
| Embedding / Chroma | 检索层有 Hashing、FastEmbed/BGE、API encoder 和 `ChromaPolicyRetriever`；在线/本地模式由 CLI 配置。 | 现有 Chroma 集合专供政策片段。经验可复用 encoder，但必须用独立集合；Hashing 是离线基线，不能声称语义等价。 |
| 失败测试 | `test_planning.py` 覆盖坏计划回退；`test_safe_execution.py` 覆盖政策证据缺失和执行安全；`test_tool_runtime.py`、`test_observability.py` 等覆盖重试、Trace。 | 尚无长期经验的资格、去重、失效、检索或重复失败率测试。 |
| Reflection / Memory | `recovery.FailureAnalyzer` 只把工具错误码映射到恢复策略；`TaskState.failure_events` 是局部执行记录；`LayeredMemoryManager` 是会话摘要/情节记忆。 | 它们都不验证 Agent 根因，也不构成可复用的 Failure Memory。现有 `FailureAnalyzer` 名称须与新分析器区分。 |

## 可可靠捕获与不可可靠判断

可由现有证据可靠识别：严格计划 Schema 拒绝及模板回退；目标/订单版本变化和旧计划取消；工具返回的确定性 `schema`、`precondition`、`authorization`、`permission`、`policy_evidence`；operation ID 对应的持久业务状态；重复副作用的 `commerce_attempts`。只有证明 Agent 选择或动作违反规则时才具备写入资格。比如计划拒绝可证明候选规划错误，但当前模板已自动修复，是否值得长期保存仍须看可复用性。

不能仅凭现有数据判断：第三方 timeout/500 的责任归属；最终回复文字是否代表任务成功；检索为空究竟是政策缺失还是索引故障；用户取消是否属于失败；无完整 Trace 时的根因。`outcome_unknown` 必须查询原 operation ID，不能把它直接写成工具失败。已有恢复策略在结果仍未知时冻结并转人工，这是正确安全行为，不应当作为负面经验。

## 关键接入约束

1. 资格判断以外部业务断言、计划验证结果和持久业务状态为证据；LLM 的反思仅生成候选经验，不能确认失败。
2. 副作用记忆只提供上下文，`PolicyGuard`、授权和 `RecoveryPolicy` 仍是执行边界。
3. Trace ID 可用于审计，但 Trace 清理后不能再依赖其内容；记录证据类型、指纹和版本，避免储存用户原文及订单个人信息。
4. 动态路径可以在规划、工具前和恢复阶段检索；非动态路径须单独设计，不可假定同一状态模型。
