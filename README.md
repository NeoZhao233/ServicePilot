# ServicePilot

面向电商售后的可控智能客服 Agent。项目不是 FAQ 聊天壳，而是在订单事实和政策约束下完成“识别诉求 → 校验权限 → 检索政策 → 判断资格 → 用户确认 → 执行业务操作”的闭环，并保留可回放的 Agent Trace。

## 已实现能力

- **受控 Agent Runtime**：LangGraph 编排请求校验、意图路由、业务执行、结果反查和安全降级；退款等敏感操作必须确认，工具成功前禁止声称完成。
- **多路意图识别**：关键词、Embedding 相似度、可选 LLM 三路加权投票，覆盖 10 类售后意图和 4 档紧急度；带置信度阈值、LRU 缓存、动态工具白名单和 Prompt Profile。
- **Hybrid RAG**：BM25 + 本地 BGE/Chroma 语义召回、Query Rewrite、并行检索、RRF 融合、DeepSeek LLM Rerank、metadata 过滤、TTL 缓存及向量/重排故障降级。
- **三层记忆**：Redis TTL 短期会话（SQLite 自动降级）、向量化情景记忆、显式白名单用户画像；按 Token 预算动态选择上下文并隔离不同用户。
- **MCP 工具服务**：订单、政策、退款、工单 4 个真实 MCP 工具；工具内部再次执行订单归属、资格和确认校验。
- **FastAPI 可观测 Web 演示**：Pydantic 请求校验、OpenAPI、会话鉴权，以及客服对话、业务状态和 Agent Trace 同屏展示。
- **用户 PDF 证据输入（可选）**：Docling 提取正文、Markdown 表格和必要图片；按用户与会话保存，提供页码引用及图片预览。图片判断只调用显式配置的视觉模型。接口、样本、20 题对照和公开 PDF 审查见 [PDF 多模态说明](docs/PDF_MULTIMODAL.md)。
- **双层可复现评测**：开发回归集用于快速防回归，另设 48 条意图、24 条检索、20 组多轮对话的真实世界挑战集；按口语、噪声、否定、复合诉求、Prompt Injection、跨订单和安全场景切片统计，并保存全部失败轨迹。

## 已验证结果

以下均为仓库内固定开发集或挑战集的实验结果，不代表生产泛化效果：

| 评测 | 结果 |
|---|---:|
| 多路意图路由（42 条） | Accuracy 97.62%，Macro-F1 97.88% |
| DeepSeek 三路意图路由（42 条） | Accuracy 100%，Macro-F1 100% |
| BM25（35 条） | Recall@3 91.43%，MRR 91.43%，NDCG@3 91.43% |
| Hybrid RRF（35 条） | Recall@3 97.14%，MRR 91.90%，NDCG@3 93.23% |
| 端到端规则/工具约束（20 组） | 20/20 通过 |
| DeepSeek LLM-as-Judge（20 组） | 总均分 4.72/5，安全性 5.00/5 |
| DeepSeek 意图挑战集（48 条） | Accuracy 97.92%，支持标签 Macro-F1 98.31% |
| 困难检索集（24 条，Rewrite Hybrid） | Recall@3 68.75%，NDCG@3 61.17% |
| BGE 语义向量（同一困难集） | Recall@3 89.58%，NDCG@3 83.28% |
| BGE + Rewrite + DeepSeek LLM Rerank | Recall@3 / MRR / NDCG@3 均为 100% |
| DeepSeek 多轮业务挑战集（20 组） | 15/20（75%） |
| 自动化测试 | 以当前测试命令输出为准 |

原始结果位于 `evaluations/results/`，挑战集设计、切片结果和 5 个保留失败项见 [真实世界挑战报告](evaluations/CHALLENGE_REPORT.md)。离线 Hashing Vector 只是无网络工程基线；接入本地 BGE 后语义向量超过 BM25，DeepSeek 只负责候选重排，不冒充 Embedding 模型。

## 快速开始

```bash
uv sync --extra agent --extra api --extra dev --extra semantic
uv run servicepilot-web --port 8765
```

打开 `http://127.0.0.1:8765`，推荐依次体验：

```text
查询 ORD-1003 的物流
我想退掉 ORD-1001
确认
ORD-1002 的耳机有质量问题
帮我查 ORD-2001 的物流
```

启用 Chroma 和 Redis（Redis不可用时自动回退 SQLite）：

```bash
uv run servicepilot-web \
  --chroma-dir .servicepilot/chroma \
  --session-db .servicepilot/sessions.sqlite3 \
  --redis-url redis://127.0.0.1:6379/0
```

模型环境变量和在线模式见 [在线配置](docs/ONLINE_MODE.md)。

DeepSeek 可直接用于意图路由和 LLM-as-Judge；由于其官方 API 未提供
Embeddings，本项目使用本地 BGE 中文语义向量，并用 DeepSeek 重排候选：

```bash
cp .env.deepseek.example .env
# 手动填写 .env 中的 DeepSeek Key，然后加载：
set -a; source .env; set +a
uv run servicepilot-web --online --port 8765 --chroma-dir .servicepilot/chroma-deepseek
```

## MCP 服务

```bash
uv run servicepilot-mcp --transport stdio
```

暴露 `get_order`、`search_policy`、`create_refund_request`、`create_ticket`。退款工具要求 `user_confirmed=true`，订单类工具都要求已认证 `user_id`。

## 测试和评测

```bash
uv run python -m unittest discover -s tests -v
uv run servicepilot-intent-eval --output evaluations/results/intent_offline.json
uv run servicepilot-retrieval-eval --output evaluations/results/retrieval_offline.json
uv run servicepilot-eval --output evaluations/results/business_offline.json

# 独立挑战集
uv run servicepilot-intent-eval --cases evaluations/intent_challenge.json --online
uv run servicepilot-retrieval-eval --cases evaluations/retrieval_challenge.json
uv run servicepilot-eval --cases evaluations/cases_challenge.json --online
```

配置在线模型后，可运行三路意图/真实 Embedding 对比和 20 组 LLM-as-Judge：

```bash
uv run servicepilot-intent-eval --online --output evaluations/results/intent_online.json
uv run servicepilot-retrieval-eval --online --output evaluations/results/retrieval_online.json
uv run servicepilot-eval --judge-online --output evaluations/results/business_judged.json
```

## 核心结构

```text
src/servicepilot/
├── graph_runtime.py       # LangGraph 受控工作流与结果验证
├── agent.py               # 售后状态机与确定性业务编排
├── routing.py             # 关键词/Embedding/LLM 投票、紧急度、LRU
├── retrieval.py           # BM25、BGE、Chroma、Rewrite、RRF、缓存与降级
├── memory.py              # SQLite/Redis SessionStore 与审计事件
├── failure_memory.py      # 验证后失败经验、长期存储与按需召回
├── layered_memory.py      # 情景记忆、用户画像、Token 预算上下文
├── mcp_server.py          # FastMCP 业务工具服务
├── tool_runtime.py        # 重试、熔断、异常隔离
├── providers.py           # 在线模型、Embedding、LLM Rerank/Judge 适配
├── evaluation.py          # 端到端评测
└── webapp.py              # Web API 与演示界面
```

架构原则：模型负责理解语言，RAG 负责提供政策候选，确定性代码负责权限、资格、金额限制和状态变更。完整边界见 [架构说明](docs/ARCHITECTURE.md)，与参考简历的逐项验收见 [能力验收表](docs/REFERENCE_GAP.md)。

## 可写入简历的版本

**ServicePilot — 电商售后智能客服 Agent 平台**  
技术栈：Python、FastAPI、LangGraph、MCP、Redis、ChromaDB、Hybrid RAG（BM25 + RRF）

- 基于 LangGraph 构建带请求校验、意图路由、业务执行、结果反查及安全降级的受控工作流；通过意图级工具白名单、敏感操作确认、订单归属校验和重试熔断约束 Agent 行为，并提供可回放 Trace。
- 实现关键词、Embedding 与 DeepSeek 的多路加权意图路由，覆盖 10 类售后意图和 4 档紧急度；在包含口语、错别字、否定、复合诉求与 Prompt Injection 的 48 条挑战集上取得 97.92% Accuracy，并定位 1 条注入攻击误路由。
- 构建 9 个业务域、37 个政策块的 Hybrid RAG，融合 BM25 与本地 BGE 中文语义向量，支持 Query Rewrite、并行召回、RRF、DeepSeek LLM Rerank、metadata 过滤、TTL 缓存和降级；在冻结的 24 条困难检索集上，语义向量 Recall@3 达 89.58%，加入候选重排后 Recall@3、MRR、NDCG@3 均为 100%。
- 设计 Redis 短期会话、向量情景记忆与白名单用户画像三层记忆，按 Token 预算动态组装上下文；实现 4 个 MCP 工具，并用 20 组多轮挑战对话验证越权、确认、幂等、跨订单和注入安全，DeepSeek 在线模式通过 15/20，保留 5 条失败轨迹作为改进基线。
- 为动态任务增加失败经验的候选留存、Trace 与业务状态核验、去重及容量受限的 LRU 召回；扩展可选意图路由经验注入。在 24 道配对的虚构客服任务中，真实模型与完整 Agent 流程的最终结果由 21/24 提升到 24/24；提升来自安全澄清任务，正常退款、物流和转人工任务两组均完成。结果仅适用于该模型与测试切片。

这些数字均可由仓库命令和原始 JSON 复现。DeepSeek 未提供 Embeddings API，因此简历中必须写“本地 BGE Embedding + DeepSeek Rerank”，不能写成 DeepSeek Embedding；24 条人工挑战集也不代表生产泛化准确率。

## 动态任务与安全恢复

新增事务化业务账本、版本绑定授权、退款超时对账，以及可选动态规划子图。
原流程保留；新旧路径均受业务层安全校验。启用示例：

```sh
.venv/bin/python -m servicepilot.webapp --dynamic-planning
```

详细配置、持久化和恢复步骤见 [运行说明](docs/DYNAMIC_EXECUTION.md)；
实际验证和实验局限见 [升级交付记录](docs/UPGRADE_DELIVERY.md)。
在同一安全底座的 12 道政策证据故障注入任务中，静态路径达成预定结果 6/12，
动态执行路径为 12/12；使用真实 DeepSeek 候选计划复核仍为 12/12。
增益来自显式政策证据前置步骤，不能单独归因于目标变化后的重新规划，见
[动态路径配对实验](evaluations/DYNAMIC_POLICY_ABLATION_REPORT.md)。

可选 Failure Memory 将验证过的规划失败提炼为长期经验，在动态规划前按需召回；
并提供 B0/B1/B2 对照路径与独立评测汇总入口。实现依据、启用方式及尚未完成的
线上效果验证见 [Failure Memory 指南](docs/memory/README.md)。

### 失败日记：3 个可复现演示

运行固定离线案例，无需 API Key：

```bash
.venv/bin/python evaluations/run_failure_memory_showcase.py \
  --output evaluations/results/failure_memory_showcase.json
```

脚本依次验证正常退款、旧计划执行拦截，以及首次计划失败后的 B0（无失败记忆）/B2（验证后召回）对比。它使用真实 LangGraph 运行时、Trace 和 SQLite 业务账本；规划模型是**脚本化固定响应**。完整逐例结果在[原始 JSON](evaluations/results/failure_memory_showcase.json)。

| 固定案例 | 实际结果 |
| --- | --- |
| 正常退款 | 确认前 0 笔、确认后 1 笔，任务完成并反查账本 |
| 旧计划故障注入 | 执行被拦截，0 笔退款；Trace 记录候选与验证，生成 1 条 `GOAL_DRIFT` 经验 |
| 3 组同类后续任务 | B0 计划校验错误 3 次，B2 为 0 次；两组均完成 3/3，重复退款均为 0 |

这个小样例证明了经验写入、注入和安全拦截可复现，不能推断真实 LLM 的错误率改善。线上模型效果仍需独立任务集评估。

**真实模型验证**：Planner 的 24 组 B0/B2 配对中，两组计划校验错误均为 **0/24**，未观察到这项指标的改善，见[规划测试报告](evaluations/FAILURE_MEMORY_REAL_REPORT.md)。转到已有真实错误的意图分类环节后，一条已验证经验使 `deepseek-flash` 在 30 道冻结角色伪装题上的正确数由 **20/30** 升至 **30/30**，10 道正常诉求两组均为 **10/10**。逐题结果、复现命令与边界见[意图分类实验报告](evaluations/FAILURE_MEMORY_INTENT_REPORT.md)。
同题集换顺序复测为 **23/30 → 30/30**，正常诉求仍为 **10/10**；两轮分别报告，不把重复题目算成新的独立样本。

进一步运行完整客服图流程，并核对工具调用与虚构订单账本：24 道合成任务首轮 **21/24 → 24/24**，换顺序复测 **23/24 → 24/24**。改善来自没有真实售后诉求时正确澄清、避免错误创建工单或进入发票流程；退款、物流和转人工任务两组均完成。见[完整任务实验报告](evaluations/FAILURE_MEMORY_TASK_REPORT.md)。

另在新的 24 道同类错误后续任务上单独测量复发率：关闭失败日记首轮 **1/24**、同题换序复测 **2/24**，开启后两轮均为 **0/24**；6 道正常业务对照两组均通过。见[同类错误复发率实验](evaluations/FAILURE_RECURRENCE_REPORT.md)。
后续用 `deepseek-flash` 对同批题做仅 B0 的首次失败探索，仍为 **0/12 次错误**。当前 Planner 的模板与校验器已覆盖这些风险，尚无真实模型失败样本可用于有说服力的“失败后学习”测试。

## Agent Trace 可观测性

现已支持请求、LangGraph、模型、混合检索、工具和动态规划的持久化执行轨迹。
启动后访问 `/traces`；查询需要配置 `SERVICEPILOT_AUTH_TOKENS`。
完整启动、鉴权、采样、清理、导出和故障排查说明见 [Trace 使用说明](docs/observability/README.md)，
设计依据见 [架构审计](docs/observability/architecture_audit.md)。

## Jev 自纠错 RAG（实验功能）

已加入可选 Jev 评分、受校准门槛控制的证据过滤、充分性判断及最多两轮定向重检索。
默认仍为原始 RAG，配置 `SERVICEPILOT_RAG_MODE` 可选择研究模式；过滤和纠错需要真实校准数据及政策来源/版本信息。
参见 [实现与运行说明](docs/research/jev_corrective_rag.md)。当前尚无 Jev 实网效果结论。

## 可插拔自纠错 RAG（DeepSeek 当前实现）

新配置及限制见 [使用说明](docs/rag/README.md)，[代码审计](docs/rag/architecture_audit.md) 和 [实验报告](docs/rag/ablation_report.md)。`SERVICEPILOT_RAG_MODE=corrective`、`SERVICEPILOT_EVALUATOR_PROVIDER=deepseek` 不需要 Jev Key。默认仍为原始检索；动态任务使用空证据转人工机制。旧 Jev 研究文档保留作历史记录，不代表本轮配置要求。
