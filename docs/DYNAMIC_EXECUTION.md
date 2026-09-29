# 动态任务与安全恢复：运行和边界

## 启动

在 servicepilot 目录使用现有虚拟环境，不需要升级依赖：

```sh
.venv/bin/python -m servicepilot.webapp --dynamic-planning
```

Web 默认监听 127.0.0.1:8000，默认数据库为 `.servicepilot/sessions.sqlite3`。网页和 API 均保留原路径。退款和换货受理进入动态子图，普通物流查询继续使用原流程。

终端模式：

```sh
.venv/bin/python -m servicepilot.cli --dynamic-planning --session-db .servicepilot/sessions.sqlite3
```

库调用 `build_agent(..., dynamic=True, commerce_db=..., session_db=...)`；未配置持久化路径时仍是隔离的内存演示。配置 commerce_db 而省略 session_db 时会使用同一路径保存会话。SQLite 会话表与业务表使用不同名称，不删除已有数据；种子订单只 INSERT OR IGNORE，重启不会覆盖退款结果。

关闭 `--dynamic-planning` 且不设置 `SERVICEPILOT_DYNAMIC_PLANNING=true`，即可回到原工作流。事务、归属、确认、幂等和未知结果恢复不会被关闭。完全原始的 B0 仅在评测归档中运行，不作为部署回退选项。

## 模型规划

默认 TaskPlanner 生成确定性受限计划。设置 `SERVICEPILOT_PLANNER_MODEL` 后使用现有 transport 调用 `/chat/completions` JSON mode；配置 API_KEY、BASE_URL、INTENT_MODEL 和 EMBEDDING_MODE=hashing。候选计划必须经过本地严格 Schema、工具白名单、参数、依赖环与安全谓词检查。不合法时只回退一次模板。

`SERVICEPILOT_INTENT_MODEL` 是现有 settings 的必填项，即便只启用候选规划也须提供。原 online 路由/RAG 仍使用 Responses 兼容协议；直接使用 DeepSeek 原生端点时，可以保持 `--online` 关闭、仅启用 PLANNER_MODEL，避免把原路由误当作原生 Chat Completions 客户端。本轮仅完成假 transport 合约测试，未使用真实密钥调用模型，不报告在线 Token 或延迟收益。

## 安全执行协议

1. 读取最新订单；金额规范化为整数分，订单具有单调 version。
2. 生成绑定用户、订单快照、金额、会话、目标/计划版本和 15 分钟有效期的授权记录。Token 只由受信运行时生成。
3. 只接受明确确认话语。新目标优先，含确认关键词的任意句子不算确认。
4. 退款前保存 operation_id，业务事务内验证授权、归属、版本和资格，再写唯一退款记录、订单状态与操作结果。
5. 超时不自动重试退款。按原操作 ID 查询；已成功则继续，无法确认则冻结并写入人工处理队列。
6. 动态执行器还会查询权威账本验证预期效果，不以“返回 ok”或模型描述作为完成依据。

本实现是 SQLite 业务模拟器，不是实际支付渠道。接入远端业务必须由远端提供幂等键、条件版本写入与权威操作查询；本地事务无法保证远端 exactly-once。只支持每订单一次全额退款申请，不支持部分退款、真实打款或库存履约。换货成功仅指核验工单创建成功。

## MCP 与身份

```sh
.venv/bin/python -m servicepilot.mcp_server --commerce-db .servicepilot/sessions.sqlite3 --user USER-001
```

MCP CLI 注册查询操作状态工具，和本地运行时共享数据库。退款工具保留 user_confirmed 参数以兼容调用形状，但仍需受信确认边界已经确认的 authorization token；模型不能通过 true 创建授权。服务不公开 mint/confirm 授权工具。

MCP CLI 将 user_id 绑定到受信连接的 `--user`；单用户连接不应用于开放的多人 HTTP MCP 服务。多用户生产接入需要对接认证网关。

HTTP 设置 `SERVICEPILOT_AUTH_TOKENS` 为 JSON 的 token→user_id 映射后，业务请求必须使用 Bearer token，且请求 user_id 必须匹配。未设置时保持原本地演示模式；不能宣称已具备生产身份认证。内置网页尚未提供登录界面，开启认证时需由网关注入凭证或使用 API 客户端。

## 持久化、并发与人工接管

- SQLite：订单、授权、操作、退款、工单、任务计划、人工处理队列、会话及审计事件。
- Redis：原会话缓存/TTL 逻辑保留，不作为退款事实来源。
- LangGraph：编排状态转换；未接入 checkpointer。跨轮恢复读取业务任务表和操作账本，不依赖图节点重放。
- 同一会话使用数据库互斥记录串行执行。并发请求返回 409；新用户目标按获取执行权的顺序处理，不能抢占已经提交的业务操作。
- 进程崩溃后互斥记录不自动过期，避免旧进程恢复后成为第二写入者。确认原进程停止后，由运维恢复：

```sh
.venv/bin/python -m servicepilot.operations --database .servicepilot/sessions.sqlite3
.venv/bin/python -m servicepilot.operations --database .servicepilot/sessions.sqlite3 --release-session SESSION-ID --process-stopped
```

恢复命令只释放会话锁，不清空授权/任务/操作账本。继续会话时先对账。人工队列是持久化待处理记录，本轮未对接真人客服通知服务，也不自动发送消息。

首期每个会话承载一个动态售后任务；已提交退款后改变目标会转人工，新独立订单售后建议创建新会话。默认最多 6 次规划、40 次业务工具预算、3 次连续失败；读取重试仍受原工具执行器限制。任务版本、错误、失效原因和计划历史均落库。

## 验证

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/python evaluations/run_dynamic_ablation.py --repeat 3
.venv/bin/python docs/research/assessment_probe.py
```

评测中关闭新安全谓词只发生在隔离子进程，生产代码没有跳过 Guard 的配置。B0/B1 没有业务数据库，结果明确保留 ledger=null。B2 比 B1 同时更换执行后端，因此不是单变量因果实验；SAFE 组才用于判断新规划的增量收益。

另有[政策证据缺失的配对实验](../evaluations/DYNAMIC_POLICY_ABLATION_REPORT.md)：使用同一安全底座，仅切换动态执行开关，12 道合成任务中静态路径通过 6/12、动态路径通过 12/12；真实 DeepSeek 候选计划复核保持相同结果。收益来自计划中的政策证据前置检查，并非目标变化后重规划的独立效果。
