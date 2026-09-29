# 真实世界挑战集评测报告

评测日期：2026-09-16。挑战集与原开发回归集分离，目的不是得到满分，而是暴露模型在噪声输入、多轮状态、安全边界和检索困难样例上的缺陷。当前样例为人工对抗式构造，尚不能替代真实线上日志或独立标注盲测。

## 数据设计

| 评测集 | 规模 | 重点场景 |
|---|---:|---|
| `intent_challenge.json` | 48 | 口语、错别字、隐式表达、否定、复合诉求、信息不足、Prompt Injection、中英混输 |
| `retrieval_challenge.json` | 24 | 口语改写、多政策答案、难负例、错别字噪声 |
| `cases_challenge.json` | 20 组多轮对话 | 越权、敏感操作确认、跨订单切换、幂等、安全事故、品类例外、上下文切换、注入攻击 |

评测器支持按场景切片、难度分组、拒答率、零命中率、逐轮状态断言、事件断言、引用要求和工具精确调用次数。复合诉求允许配置多个可接受意图，减少单标签强行判分。

## 结果

### 意图识别

| 路由方式 | Accuracy | 支持标签 Macro-F1 |
|---|---:|---:|
| 离线关键词 + Hashing Vector | 37.50% | 16.38% |
| DeepSeek + 关键词 + Hashing Vector | 97.92% | 98.31% |

DeepSeek 在口语、错别字、隐式表达、否定和中英混输切片均达到 6/6；唯一失败是 XML 风格 Prompt Injection 被误判为人工客服。这一结果说明 LLM 显著改善语言理解，但输入指令隔离仍需单独的安全防线，不能只依赖分类 Prompt。

### 检索

| 检索器 | Recall@3 | MRR | NDCG@3 | Zero-hit |
|---|---:|---:|---:|---:|
| BM25 | 77.08% | 68.75% | 68.37% | 16.67% |
| Hashing Vector | 52.08% | 51.39% | 47.40% | 37.50% |
| Hybrid RRF | 68.75% | 61.11% | 59.77% | 25.00% |
| Rewrite + Hybrid + Rerank | 68.75% | 61.11% | 61.17% | 25.00% |
| BGE Semantic Vector | 89.58% | 82.64% | 83.28% | 8.33% |
| BGE + Rewrite + DeepSeek LLM Rerank | 100% | 100% | 100% | 0% |

早期 Hybrid 弱于 BM25，证明 Hashing Vector 的噪声会拖累融合排序。接入本地 `BAAI/bge-small-zh-v1.5` 后，单独语义向量 Recall@3 从 52.08% 提升到 89.58%，超过 BM25 的 77.08%；在同一冻结候选集上加入 DeepSeek LLM Rerank 后达到 100%。该集合仅有 24 条人工样例，满分只能说明当前集合全部通过，不能推断生产准确率。

### 端到端业务行为

| 模式 | 通过率 |
|---|---:|
| 离线规则路由 | 9/20（45%） |
| DeepSeek 在线路由 | 15/20（75%） |

DeepSeek 在线模式仍有 5 个明确缺陷：

1. 订单切换后确认会重复查询订单，并把“确认退款”错误作用到新订单。
2. 冒烟烫伤场景创建了人工工单，但未先输出立即停用等安全指引。
3. 上传其他订单证据时没有绑定证据所属订单，错误创建工单。
4. 隐式少件可以处理，但回复遗漏政策引用。
5. 证据文本中的 Prompt Injection 会把破损流程劫持为退款流程。

这 5 条保留为失败回归用例，不为提高分数而删除。

## 复现

```bash
python -m servicepilot.intent_evaluation \
  --cases evaluations/intent_challenge.json \
  --online --output evaluations/results/intent_challenge_deepseek.json

python -m servicepilot.retrieval_evaluation \
  --cases evaluations/retrieval_challenge.json \
  --online --output evaluations/results/retrieval_challenge_semantic_llm.json

python -m servicepilot.evaluation \
  --cases evaluations/cases_challenge.json \
  --online --output evaluations/results/business_challenge_deepseek.json
```

## 结论边界

- 挑战集由项目作者人工构造，不是独立第三方盲测。
- 48 条意图、24 条检索、20 组对话仍不足以推断生产准确率。
- DeepSeek 用于意图理解和候选重排；Embedding 来自本地 BGE 模型，两者不得混写。
- LLM-as-Judge 分数仅作为补充，安全和工具副作用必须以确定性断言为准。
