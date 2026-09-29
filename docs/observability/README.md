# ServicePilot Agent Trace 使用说明

每次 HTTP、直接 Agent 调用或 MCP 服务端调用生成独立 Trace。`trace_id` 和 `request_id` 独立生成；`thread_id` 关联会话，`task_id` 关联业务任务（当前业务模型中任务 ID 等于会话 ID）。计划版本与目标版本独立记录。Trace 只观察执行，订单、退款与授权状态仍以业务数据库为准。

## 启动与查看

在项目根目录使用现有虚拟环境：

```sh
export SERVICEPILOT_AUTH_TOKENS='{"your-local-secret":"USER-001"}'
export SERVICEPILOT_TRACE_ENABLED=true
.venv/bin/python -m servicepilot.webapp --dynamic-planning --session-db .servicepilot/sessions.sqlite3
```

打开 `http://127.0.0.1:8000/traces`，输入配置的凭证后查询。凭证只保留在当前页面内存中，不写入浏览器持久存储。未配置身份认证时，业务演示仍能运行，但轨迹查询拒绝匿名访问。生产部署继续使用现有身份映射和 HTTPS；不新增绕过鉴权的管理员入口。

可通过客服页面的“查看执行轨迹”入口进入。聊天响应包含 `trace_id`，HTTP 响应头包含 `x-trace-id`、`x-request-id`。未采样请求不会产生记录。带认证的 API 调用示例（`TOKEN` 由使用者自行设置）：

```sh
curl -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"user_id":"USER-001","message":"退掉 ORD-1001"}' http://127.0.0.1:8000/api/chat
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/traces
```

页面提供分页列表、状态/时间/任务/会话/节点/异常筛选、可展开执行树、节点脱敏详情、并行时间线、计划变化与恢复事件、指标及 JSON 导出。切换凭证立即清空已加载内容。

## 查询与故障定位

- `GET /api/traces`：`limit`（1–100）、`offset`、`start/end`（Unix 秒）、`status`、`thread_id`、`task_id`、`node`、`error_type`。
- `GET /api/traces/{trace_id}`：Trace、扁平 Span、父子树、结构化事件。
- `GET /api/traces/{trace_id}/metrics`：耗时、Token、调用、重规划、异常计数。
- `GET /api/tasks/{task_id}/traces` 和 `/api/threads/{thread_id}/traces`：跨请求关联。
- `GET /api/traces/{trace_id}/export`：相同权限范围内的 JSON 下载。

所有查询按已认证用户的 HMAC 身份在 SQL 层过滤；其他用户的 ID 与不存在的 ID 均返回 404。私有响应禁止缓存。存储不可用返回 503，不返回数据库异常文本。

排查失败时，先看 Trace 状态和异常 Span，再展开失败工具的父节点；同一次敏感操作的“工具响应失败”与“业务已提交”可能同时成立。沿 `operation_id`、恢复查询和 `RECOVERY_COMPLETED` 判断最终结果，不能仅凭超时重复退款。查看 `PLAN_INVALIDATED`、`REPLAN_TRIGGERED` 和后续 `PLAN_CREATED` 的旧/新版本、保留与取消步骤。`dynamic.plan` 的属性提供重规划耗时及其实际子模型调用的额外 Token。

`ok` 表示请求正常结束，不表示没有恢复过的错误；业务结果另见 `outcome`。`error_count` 是失败 Span 数，包含向父节点传播的异常，不等于不同事故数。工具计数使用实际尝试及独立查询，排除嵌套存储查询的重复计数。

## 采集与数据治理

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `SERVICEPILOT_TRACE_ENABLED` | true | 总开关 |
| `SERVICEPILOT_TRACE_SAMPLE_RATE` | 1 | 0–1，根请求只做一次采样 |
| `SERVICEPILOT_TRACE_DETAILS` | false | 开启后仅增加安全领域词预览，不保存原始敏感文本 |
| `SERVICEPILOT_TRACE_RETENTION_DAYS` | 14 | 清理命令默认保留期 |
| `SERVICEPILOT_TRACE_DB` | 商务 SQLite | 可使用独立文件，目录需存在 |
| `SERVICEPILOT_TRACE_HMAC_KEY` | obs_meta 内持久化密钥 | 可显式配置；多进程需一致 |

自由文本默认仅保存长度和带密钥摘要；用户/订单 ID 伪名化，支付金额、地址、联系方式、授权、密钥及隐藏推理字段不采集。政策正文保留文档 ID、内容版本摘要和得分。不要把详情模式当作原始 Prompt 日志。更换 HMAC 密钥会改变身份映射，旧数据不再匹配新身份摘要；保留数据库及密钥的受控备份。

Trace 开始时写入 running；结束时一次事务写入汇总、Span 和 Event。进程异常退出可能只留下 running 记录，不能凭空恢复未刷盘的子 Span。写入失败记录固定告警和内存计数，放弃该次持久化，不重试业务操作。独立 SQLite 连接、50 ms 锁等待上限；每条最多 1000 Span、2000 Event，超过上限标记 incomplete，相关指标未知。无后台自动清理任务，按运维计划执行：

```sh
.venv/bin/python -m servicepilot.trace_admin --database .servicepilot/sessions.sqlite3 --cleanup
# 确认所有应用 worker 已停止后，才修正遗留 running 记录：
.venv/bin/python -m servicepilot.trace_admin --database .servicepilot/sessions.sqlite3 \
  --mark-abandoned --workers-stopped
```

清理仅删除过期且已结束的 obs_* 记录，通过外键级联，不删除订单、退款、会话或业务事务日志。

## 指标、导出与边界

模型用量取实际 API 返回值，缺失为 null/“未知”；没有 LLM 调用时计数与 Token 合计为 0。模型类型、耗时、缓存和费用只报告可获得数据，不估算费用。embedding 单独分类，LLM 汇总不重复计入 embedding。并行 Span 的时长不能相加当作总耗时。

导出 JSON 可用于按任务聚合重规划率、恢复率、失败类型、耗时和已知 Token 的 Agent 评测。保持 null 的未知语义，并排除 incomplete 记录或另报覆盖率；脱敏数据不能重放原始用户对话。当前没有外部遥测发送器、OTel/LangSmith 双写或原始推理存储。

当前 Agent 直接使用本地工具，记录为 tool_attempt；实际 MCP 服务入口记录为 mcp，不将本地调用冒充网络调用。跨独立 MCP 进程尚无自动父 Trace 传播，当前项目也没有 MCP 网络客户端。传入 HTTP traceparent 只作为上游关联保存，不能覆盖本地随机 ID 或访问身份。流式结束与取消已通过真实 StreamingResponse 测试，现有聊天产品本身仍是非流式接口。

## 验证与性能

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts/benchmark_trace.py
PYTHONPATH=src .venv/bin/python scripts/benchmark_trace.py
```

新增 18 项测试检查真实 SQLite 记录：父子树、并行 RAG、异步与线程隔离、用量/缺失用量、实际 MCP 调度成功与拒绝、响应丢失后的业务恢复、存储写失败、隐私、鉴权、422、流式结束/断连/取消、保留清理、停机遗留记录和采样。模型通过 Mock transport 测试，MCP 通过内存 FastMCP 调度现有本地工具测试；不调用真实模型或支付渠道。

2026-09-24 本机 Python 3.11，离线物流请求，FastAPI TestClient，动态规划开启，同一 SQLite 文件存商务与观测数据。每组预热 5 次、交替执行 50 次，初始化不计入耗时：

| 追踪 | 平均 ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: |
| 关闭 | 4.475 | 4.408 | 4.873 |
| 开启 | 6.095 | 6.028 | 6.585 |

平均增加 1.620 ms（约 36.2%）。这是快速离线请求的本机结果，不能推断真实模型网络、并发高负载、远端 MCP 或长期大数据库的性能。原始结果见 benchmark.json。浏览器实测已验证授权列表、执行树、指标、计划事件与脱敏详情正常显示。
