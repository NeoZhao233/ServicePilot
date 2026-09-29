# 可插拔证据评估与有界纠错

当前 provider 默认 DeepSeek，Jev 是未来适配边界，不要求 Jev SDK 或 Key。系统总体默认模式仍是 original，避免未实测即替换原始 Rerank。

## 配置

使用现有 `SERVICEPILOT_API_KEY`、`SERVICEPILOT_BASE_URL`、`SERVICEPILOT_INTENT_MODEL`。请在启动进程环境中加载配置，CLI 不会自动加载 `.env`。

```sh
export SERVICEPILOT_RAG_MODE=corrective
export SERVICEPILOT_EVALUATOR_PROVIDER=deepseek
export SERVICEPILOT_EVALUATOR_FALLBACK=false
export SERVICEPILOT_EVALUATOR_TIMEOUT=30
export SERVICEPILOT_EVALUATOR_CONCURRENCY=2
export SERVICEPILOT_EVALUATOR_RETRIES=0
export SERVICEPILOT_EVALUATOR_MAX_OUTPUT_TOKENS=8192
export SERVICEPILOT_RAG_MAX_CORRECTIONS=2
servicepilot --online --dynamic-planning
```

| 实验 | RAG_MODE | 当前行为 |
| --- | --- | --- |
| B0 | original | 原始 Hybrid + DeepSeek Rerank |
| B1 | evidence_score | Hybrid + 统一相关性评分，不设过滤阈值 |
| B2 | evidence_check | B1 + 四态充分性判断，无补检索 |
| B3 | corrective | B2 + 最多两次条件式补检索 |
| B4 | corrective，provider=jev | 尚未实现；明确报配置错误 |

Jev 不可用时，仅 `SERVICEPILOT_EVALUATOR_FALLBACK=true` 才回退 DeepSeek，并发出 EVALUATOR_PROVIDER_FALLBACK。构造阶段与实际决策调用均发出该事件，后者随请求 Trace 持久保存。运行时每个模型适配 Span 和判定事件始终记录实际 provider。`mock` 要求显式注入，环境变量本身不能启用虚假评分。

## 代码接口

`evaluators.py` 定义 Pydantic 输入、ScoreResult/SufficiencyResult 与 EvidenceEvaluator Protocol。评分明确返回 policy_id、score、direct_evidence、reason；充分性返回 covered_questions、missing_information、supporting_ids、retrieval_targets、diagnostic。缺失/重复/外部 ID、非有限分数、类型错误及不一致的 SUFFICIENT 结果均拒绝。

```python
from servicepilot.evaluators import (
    EvaluatorConfig, EvaluatorFactory, EvaluatorRanker, ProviderSufficiencyEvaluator,
)
from servicepilot.corrective_rag import CorrectivePolicyRetriever

provider = EvaluatorFactory.create(
    EvaluatorConfig(provider="deepseek"),
    transport=existing_reranker.transport,
    model=existing_reranker.model,
)
rag = CorrectivePolicyRetriever(
    existing_hybrid_without_reranker,
    EvaluatorRanker(provider),
    ProviderSufficiencyEvaluator(provider),
    trusted_chunks=loaded_policy_chunks,
    max_corrections=2,
)
# context 是 EvidenceContext，包含原始问题和必要商品/日期条件，不包含订单对象。
result = await rag.retrieve(query, context=context, limit=5)
```

现有同步 search 保持兼容。模型使用同一个 Responses transport；后台线程适配异步调用，限定并发，默认零重试。超时/取消立即返回异步调用者，但无法中断 urllib 的进行中网络 I/O；并发租约由实际 I/O 完成后释放，避免超额并发。配置 transport 的网络超时仍然必要。输出上限可通过 `SERVICEPILOT_EVALUATOR_MAX_OUTPUT_TOKENS` 在 512–8192 范围配置；推理模型的推理 token 可能占用此额度。显式未完成的响应即使包含可解析 JSON 也拒绝，不自动重试。本次在线实验使用 65 秒评估超时、60 秒传输超时及 8192 输出上限，不能把这些配置下的结果当成默认 30 秒配置的延迟保证。

## 规则与安全降级

只信任启动时载入的本地政策对象快照，外来文本必须满足明确来源/版本/日期条件。快照信任不补造政策版本和生效日期：模型接收 null，已明确的过期日期、类别不匹配与并行有效版本冲突仍按规则拒绝。模型输出是证据判断，不是退款授权；商务工具及归属校验未修改。`evidence-v2-policy-answer` 明确判断能否给出条件化政策解释及下一步指引，订单号、地址、支付流水、归属校验等后续执行材料不被当作检索证据缺失。只有用户要求确定个案结论、且缺少的商品或期限条件会改变结论时，才判为 UNCERTAIN。

INSUFFICIENT 根据四类有界检索目标使用既有 Hybrid 重查。无新有效 Top K、重复查询、预算耗尽、UNCERTAIN 或 CONFLICTING 均停止，返回空证据。动态任务沿用人工核对；部分静态旧路径没有安全空证据处理，所以同步兼容层回到原始检索并记录 RAG_DEGRADED。若需要完整纠错，请启用动态任务或使用异步 retrieve 并处理四态结果。原系统最终答复为业务模板，本轮没有擅自重做统一生成器。

分数是模型评分，不能当作校准概率。当前不做阈值删除；未来过滤必须用独立标注集按 provider、模型版本、prompt/schema 版本、语料指纹分别校准，不能把 DeepSeek 阈值移给 Jev。

## Jev 后续接入

`JevEvidenceEvaluator` 目前明确抛出不可用错误。将来只需实现两个 Protocol 方法、将服务响应转换成统一 Pydantic 结构、验证 ID 映射和实际用量，再完成 provider 独立校准和同样例回归。此前的 `jev.py` 和 jev_* 配置仅保留为历史研究实验，不是此工厂的已完成 Jev 适配器。

官方资料确认 Python SDK 有异步客户端及 Choice/Score/Noul 结构，但当前不能注册，因此本轮不做真实 Jev 请求。[TypeSafe 官方文档](https://docs.typesafe.ai/sdk/python/)。第三方 jev-reranker 不等于官方 SDK。

## CRAG 适配

CRAG 通过检索评估触发不同纠正动作，并采用外部检索及证据提炼。这里借鉴评估→条件式纠错，使用售后领域的四态判定及本地有界补检索；不训练原论文模型，不把互联网内容作为店铺授权政策，也不声称完整复现论文。[论文](https://arxiv.org/abs/2401.15884)、[作者代码](https://github.com/HuskyInSalt/CRAG)。
