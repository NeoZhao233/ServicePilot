# Agent Trace 实施记录

| 阶段 | 主要文件 | 实际结果与验证 |
| --- | --- | --- |
| 1 审计 | architecture_audit.md | 核对原有本地工具、原生 HTTP 模型适配、并行检索、动态子图、事务数据库及认证；采用单套本地 Trace/Span/Event 模型 |
| 2 自动埋点 | observability.py、trace_instrumentation.py、graph_runtime.py、providers.py、retrieval.py、tool_runtime.py、mcp_server.py | ContextVar 上下文、线程显式传播、统一节点包装、真实调用计时、API 实际用量与错误类型；并行、模型 Mock、MCP 内存调度测试通过 |
| 3 持久化与接口 | trace_store.py、trace_http.py、auth.py、api.py、webapp.py、trace_admin.py | obs_* 独立表与索引、完成时批量事务、写故障不影响业务、用户权限过滤、分页与导出；持久化故障、跨用户拒绝、保留清理测试通过 |
| 4 动态规划 | dynamic_runtime.py、planning.py、recovery.py、repository.py、trace_instrumentation.py | 目标变化、计划旧新版本、步骤复用/失效、真实恢复状态、重规划子调用 Token；跨请求改目标、响应丢失后的单次退款测试通过 |
| 5 查看页面 | web_assets/traces.html、traces.css、traces.js、index.html、app.js | 沿用原生前端；列表/树/时间线/计划/指标/导出；本机浏览器验证加载真实持久化轨迹，凭证切换防旧响应回填 |
| 6 测试与性能 | tests/test_observability.py、scripts/benchmark_trace.py、benchmark.json、README.md | 新增 18 项测试；全量 127 项通过，Ruff 通过；离线 HTTP 平均增加 1.620 ms |

未覆盖线上模型供应商与远程 MCP 实网联调，现有项目没有 MCP 网络客户端。没有新增流式聊天产品接口；生命周期能力由真实 StreamingResponse 测试验证。没有实时 Span 推送，结束前详情可能只有 running Trace；进程崩溃后无法重建未刷盘 Span。未引入跨进程分布式 Trace 导出或自动定时清理。性能数字只覆盖本机离线单请求场景。

完整操作、字段语义和限制见同目录 README.md。当前测试依赖中的 Starlette/AnyIO 有一条弃用警告，不影响本次结果。
