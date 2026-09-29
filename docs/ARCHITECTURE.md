# ServicePilot Architecture

## Request lifecycle

```text
START → validate_request → route_intent → execute_business → validate_result → END
             │                 │                 │
             └─────────────────┴── error ────────┴→ safe_fallback → END
```

LangGraph 负责显式节点和条件边；`ServiceAgent` 保留确定性业务状态。路由节点输出意图、置信度、四档紧急度、Prompt Profile 和最小权限工具白名单。执行节点只能通过 `ToolExecutor` 调用工具，结果节点反查“退款已提交/工单已创建”等声明是否存在成功工具事件，否则返回不改变订单状态的安全降级回复。

## Retrieval

政策检索对原始查询及规则改写查询并行执行 BM25 和本地 BGE/远程 API 语义向量召回。候选按 RRF 融合并按 policy ID 去重，在线模式再交由 DeepSeek 在候选范围内执行 LLM Rerank；重排异常时自动退回融合顺序。向量源异常时保留 BM25 结果；相同查询在 TTL 内命中有界缓存。Chroma 按 Embedding 模型隔离持久化 collection，避免不同向量维度污染索引，metadata 过滤在返回前再次执行。

## Memory

- 短期：Redis TTL 会话与事件；同步写入 SQLite fallback，Redis 异常后自动降级。
- 长期：每轮对话压缩为情景摘要并写入按用户隔离的向量索引。
- 用户画像：仅 `locale`、`preferred_contact`、`shoe_size` 三个显式字段可写。
- 上下文：画像和相关情景优先，历史消息由近到远按 Token 预算选择。

## Trust boundaries

| Component | May decide | Must not decide |
|---|---|---|
| LLM / intent routes | Intent, urgency, language interpretation | Order ownership, refund success |
| Retriever | Relevant policy candidates | Final eligibility |
| Rule layer | Eligibility and escalation | User intent |
| Repository / MCP tools | Business state changes | Natural-language success claim |
| Result validator | Whether claims match tool events | Fabricate missing tool output |

所有历史读取均校验用户与会话归属；退款工具同时校验用户确认、订单归属和资格。审计事件按会话、轮次和事件序号持久化，可由 Web Trace 回放。
