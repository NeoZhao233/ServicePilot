# Agent Trace 架构审计与设计

审计基于 2026-09-24 当前代码，在 Trace 改造前完成。

| 层级 | 现有实现 | 复用与缺口 |
|---|---|---|
| HTTP | webapp.py `/api/chat`，api.py `/chat`，同步 FastAPI 路由；无业务 StreamingResponse | 保留接口。新增纯 ASGI 生命周期中间件，覆盖验证失败、同步线程池、异步/流结束及断开，而不是响应对象创建时结束 |
| 图 | graph_runtime.py 五节点主图，可选 dynamic_task；dynamic_runtime.py 有 observe/reconcile/plan/select_step/execute_step/evaluate/recover/finish 子图 | 统一节点包装器，注册节点时自动创建 span；保留条件边。没有 checkpointer，任务从 SQLite 恢复 |
| 模型 | providers.UrllibJsonTransport，Responses 路由/重排和 Embeddings；planning.ChatCompletionPlannerModel 调用 Chat Completions JSON mode | 原生 HTTP，无 LangChain 模型包装。追踪 transport 的实际请求及 usage，并独立记录解析/候选计划验证结果；不采集隐藏推理 |
| 检索 | AdvancedHybridPolicyRetriever：QueryRewriter、ThreadPoolExecutor 并行 BM25/向量召回、RRF、启发式/LLM 重排、TTL 缓存；另有简单 Hybrid | 复用检索器，逐阶段 span；每次 submit 使用独立 copy_context，不能让同一个 Context 并行运行。诊断从当次执行取得，不依赖共享 last_diagnostics |
| 工具 | ToolExecutor 对实际本地工具调用重试/熔断；repository 为 SQLite 模拟服务；MCP Server 导出同一服务 | 当前没有 MCP Client 网络链路，不能把本地调用伪装成 MCP 网络请求。分别标记 local_tool 与 mcp；逐次尝试记录原异常和业务结果查询 |
| 存储 | memory.py 会话/AgentEvent 表与 Redis 缓存；repository.py commerce_* 业务表 | 复用 SQLite 文件，新建 obs_* 表及索引；独立连接/事务，Trace 不替代业务账本 |
| 现有追踪 | AgentEvent 与 agent_events 是扁平业务事件，无 span、计时、token；未集成 LangSmith/OpenTelemetry | 保留原事件及 UI，用中央事件桥和节点包装扩展，采用一种自定义 Trace 模型，无第二套 exporter |
| 前端 | 原生 HTML/CSS/JS，web_assets/，右侧 Agent Trace 扁平事件面板 | 在现有前端加独立 `/traces` 页面及入口，复用原生技术，不引入框架 |
| 鉴权 | auth.py 可选 token→user_id，未配置时业务仍为本地演示模式 | Trace API 始终需要已认证身份，以 HMAC 化 user_id 过滤 SQL。不能仅相信查询参数 user_id；不提供未授权管理员旁路 |
| 测试 | pytest/unittest；改造前 109 项测试；SQLite 临时库、Mock 工具/模型 | 增加真实持久化树、异步/线程隔离、故障、脱敏、鉴权与性能测试 |

## 技术设计

- Trace 对应一次 HTTP/CLI/MCP 请求。trace_id 与 request_id 独立生成；外部 traceparent 只记录关联，不覆盖本地 ID，避免跨用户碰撞。
- thread_id 对应 SessionState.session_id。task_id 对应已存在 TaskState.task_id（现版本业务恰好使用会话 ID 作为任务 ID；Trace 字段仍分离，不能把 trace_id 当会话 ID）。普通查询 task_id=null。
- ContextVar 保存当前 collector 与父 span ID；同步/异步装饰器和图节点包装自动传播。共享 collector 使用锁，跨请求对象绝不复用。
- trace/span/event 写入同一模型。持久化采用起始 Trace + 请求结束批量落库；异常/取消同样 finalize。短超时、失败吞掉并记录固定诊断计数，不覆盖业务异常；已有业务事务不依赖 Trace 提交。
- schema 以独立 obs_* 表幂等创建，旧业务表无需迁移。running Trace 崩溃后可按阈值标记 interrupted；清理只删已结束 obs_* 数据。
- 默认不存自由文本、完整消息、模型 prompt/正文、隐藏推理、凭证、订单金额/地址。结构化字段白名单、自由文本摘要哈希、ID 伪名化；详细模式仍执行脱敏，不能关闭隐私过滤。
- Token 按真实 provider usage 汇总；任一实际 LLM 调用缺少某项用量，该项总量为 null。没有 LLM 调用时零是实测值。cache/cost 缺失使用 null，不估算费用。
- HTTP 状态表示请求生命周期，业务完成/待确认/人工接管另存 outcome，不能把 HTTP 200 当作退款成功。
