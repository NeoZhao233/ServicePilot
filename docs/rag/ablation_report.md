# 可插拔决策 RAG：在线对照结果

实验完成于 2026-09-25。B0 是原始 Hybrid + DeepSeek Rerank；B1 是 Hybrid + DeepSeek 证据评分；B2 在 B1 上增加证据充分性判断；B3 再增加有界纠错检索。B4 是尚未接入的 Jev。此前 `jev_corrective` 目录的同名组别属于另一轮实验，不能混用。

## 基线与样本

改动核心行为前，使用四个原始源码快照隔离重跑冻结的 24 条问题，原始输入 SHA256 未变。离线 Hashing Hybrid + Rewrite 启发式 Recall@5=0.6875、MRR=0.6111，见 `evaluations/results/pluggable_rag/original_offline.json`；这不是模型在线基线。

原 B0 在线基线是 24 条 BGE + DeepSeek Rerank，Recall@5=1.0、MRR=0.9583，原始观测保存在 `evaluations/results/jev_stage1/b0_online.json`。本轮比较同一冻结集的初始候选，额外用 7 条复杂售后案例检查普通、特殊商品、多政策、缺失、版本冲突、过期和噪声。版本冲突及过期政策是明确标记的测试 fixture，未写入生产政策。

## 实际请求与失败

用户明确授权把冻结测试问题和本地政策文本发给已配置的 DeepSeek，最多剩余 95 次、零重试。本轮实际 81 次请求、81 个 HTTP 响应；服务端报告输入 133883、输出 225842 token，其中输出包含推理 token。模型调用平均 12.353 秒、中位 10.726 秒、p95 24.240 秒。这只是实际 HTTP 调用耗时。API 未返回价格，成本为 `null`。更早一次沙箱内失败的网络尝试另存于 `live.json`，没有可核实的服务端用量，不并入上述 81 次。

三次失败均保留，没有重试或填补预测：

| 案例与阶段 | 已确认的结果 | 处理 |
| --- | --- | --- |
| `para_01` B2 | 4096 输出 token 全部用尽，后续结构校验失败；旧日志未记录完整截断状态 | 停止该批次、保留失败 |
| `multi_01` B2 | 收到完成的响应，但适配器拒绝结果；旧日志未保留更细的拒绝原因 | 停止该批次、保留失败 |
| `crag_quality_exception` B2 | 服务端明确标记 `incomplete/max_output_tokens`；8192 输出 token 全部为推理 token | 停止该批次、证据未放行 |

`para_01` 的截断是由用量到上限推断，未伪称为服务端显式状态；第三个案例有明确的服务端截断状态。随后补充了截断诊断、可配置的输出上限，以及对政策回答与售后执行材料的区别。实验脚本按批次记录源码指纹，被测代码副本位于 `evaluations/baselines/policy_answer_live` 和 `evaluations/baselines/policy_answer_safety_live`。汇总保持这些版本分开，避免用前面看到的案例调试后再把它们当作未见样本。

## 冻结困难集：同配置 17 条

下表只含修正评估指令后、B0–B3 四组都完成的 17 条冻结问题。B0 从原在线 24 条中选取对应案例；其余 B1–B3 均来自同版 DeepSeek 评估器。完整原始样本没有删除。

| 组别 | Recall@5 | MRR | 逻辑 DeepSeek 调用/题 | 组合的 RAG 耗时均值 |
| --- | ---: | ---: | ---: | ---: |
| B0 | 1.0000 | 0.9706 | 1 | 12.309 秒 |
| B1 | 1.0000 | 0.9706 | 1 | 12.010 秒 |
| B2 | 1.0000 | 1.0000 | 2 | 18.300 秒 |
| B3 | 1.0000 | 1.0000 | 2 | 18.300 秒 |

17 条 B2 均判断证据充分，B3 未触发额外检索，因此 B2/B3 的结果相同。冻结集没有充分性标签，不能计算这 17 条判断的准确率。B0 的候选搜索来自历史在线运行，B1–B3 则复用保存的初始候选；组合耗时不是独立计时的聊天端到端耗时，不能据此声称新路径更快。B2 在评分前执行元数据规则，评分候选可能与 B1 不同；MRR 差异不能单独归因于充分性判断。B1–B3 共享完全相同输入的已测前缀，表中的逻辑调用和 token 不能相加当作账单。

## 新增复杂集：7 条

B1 完成 5/7 条评分；两个纯元数据 fixture 不适用于单纯评分。B2、B3 各完成 6/7 条充分性判断，已完成的 6 条与标注状态一致；按全部 7 条计，完成且判断正确为 6/7。失败的一条是上述 `crag_quality_exception`，不能当作“已判断正确”。已完成且标注关键政策的案例保留率为 1.0；该失败案例不在此分母内。未见经模型阈值误删的记录；本实现目前没有启用评分阈值删除。

`crag_missing` 在 B2 判为 INSUFFICIENT；B3 进行了初次检索加两次针对性补检索，仍未取得足够证据，停止并返回空证据。版本冲突得到 CONFLICTING；过期条款判为 INSUFFICIENT，均没有放行。B3 六个完成案例的平均检索轮数为 1.5，最多 3 轮。

当前业务答复沿用已有模板，实验没有新增统一生成器，也未对最终回答做人工或独立评审。因此最终回答正确率和无依据政策承诺次数均为 `null`。真实 API 费用同样为 `null`；这些值不可由检索召回率或 token 数推测。上述复杂样本的实际预测、轮数、失败和所有已知用量见 `evaluations/results/pluggable_rag/ablation_status.json`。

## 复现与解释

原始 HTTP 观测依次为 `live_network.json`、`live_remaining.json`、`live_policy_answer.json`、`live_safety_cases.json`。汇总器纯本地运行，不发送网络请求：

```sh
PYTHONPATH=src .venv/bin/python scripts/summarize_pluggable_rag.py \
  --run evaluations/results/pluggable_rag/live_network.json \
  --run evaluations/results/pluggable_rag/live_remaining.json \
  --run evaluations/results/pluggable_rag/live_policy_answer.json \
  --run evaluations/results/pluggable_rag/live_safety_cases.json \
  --output evaluations/results/pluggable_rag/ablation_status.json
```

报告同时保存完整冻结集的各组覆盖、同配置共同案例、复杂集每状态的分母、明确误删项、调用数、token、延迟分布和失败。缺失指标保持 `null`。模型临时性及样本量限制了任何质量提升结论；上线配置仍需按真实业务目标决定。
