# AgentSpec 适配

材料：[论文第 3–5 节](https://arxiv.org/html/2503.18666v1)、[官方仓库说明](https://github.com/haoyuwang99/AgentSpec)。已核对 trigger/check/enforce 和集成方式。尝试获取 README 所指核心执行器源码失败（路径请求未返回文件）；未完成逐文件实现核验，也未运行其依赖栈。

论文贡献：通过事件、谓词和执行约束拦截不安全行动，并评估跨领域约束效果及开销。其通用规则框架不能自动赋予业务数据库事务或远端幂等语义。

现有对应：ToolAccessPolicy 白名单、订单归属校验、Agent 二次确认和 MCP user_confirmed 检查。此前检查散落在不同入口。

工程适配：PolicyGuard.authorization_valid 在 SQLiteCommerceRepository 的同一写事务内执行。本地工作流、动态子图和 MCP 都调用该仓库；MCP 布尔确认不能自行产生授权。任务版本、订单快照、金额、有效期与确认状态必须同时满足。

自主设计：持久化授权 token、操作 ID、唯一约束、事务条件校验、可信连接身份与可选 HTTP bearer 身份边界。所有安全限制在关闭动态规划时仍保留。

未采用：DSL、ANTLR、任意自定义执行代码或 LLM 自我检查作为放行条件。固定业务谓词更容易验证，且不需要升级原依赖。

对应验证：test_guard_rejects_stale_or_foreign_authority、test_direct_call_cannot_bypass_confirmation、test_two_connections_cannot_double_refund、test_atomic_rollback_on_database_failure、test_mcp_uses_shared_ledger_and_does_not_trust_boolean、HTTP 身份伪造测试。
