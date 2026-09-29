# Development baselines

这些结果用于开发期回归，不是最终简历指标。数据由项目作者编写，规模较小，尚未经过独立标注或盲测。

## Intent routing baseline

运行命令：

```bash
PYTHONPATH=src python3 -m servicepilot.intent_evaluation
```

数据集：`evaluations/intent_cases.json`

| Metric | Result |
|---|---:|
| Cases | 24 |
| Accuracy | 0.9583 |
| Macro-F1 | 0.9577 |

唯一失败案例：

```text
输入：这双鞋我不想要了
期望：refund
预测：unknown
```

该案例保留为基线失败，不通过添加“答案关键词”的方式修复。下一阶段应验证 LLM 路由或语义相似度能否覆盖这类表达，并观察它们是否会增加未知请求的误分类。

## Business-flow seed suite

运行命令：

```bash
PYTHONPATH=src python3 -m servicepilot.evaluation
```

当前 6/6 通过。它只覆盖主流程、退款确认、越权访问、证据收集和高金额转人工，用途是持续集成回归，不代表真实客服任务成功率。

## Policy retrieval development baseline

运行命令：

```bash
PYTHONPATH=src python3 -m servicepilot.retrieval_evaluation
```

当前 8 条人工编写查询、`K=3` 的结果：

| Retriever | Recall@3 | MRR | NDCG@3 |
|---|---:|---:|---:|
| BM25 | 1.0000 | 0.8125 | 0.8616 |
| Hashing Vector | 1.0000 | 0.9375 | 0.9539 |
| Hybrid RRF | 1.0000 | 0.8125 | 0.8616 |

当前 Hybrid RRF 没有超过单独的 Hashing Vector，因此不能宣称融合检索带来了提升。Hashing Vector 只是无外部依赖的工程基线，也不能称为语义 Embedding。后续需要接入真实 Embedding、扩大测试集，并在固定测试集上重新比较。

## Before publishing metrics

正式结果至少需要：

1. 扩充到 200 条以上、多名标注者参与的数据集。
2. 固定开发集后，不再根据测试集补规则。
3. 对比 rule、LLM、embedding 与 fusion 四种策略。
4. 报告每类 Precision、Recall、F1，以及推理成本和延迟。
5. 保存模型名称、参数、Prompt 版本和原始预测，确保实验可复现。
