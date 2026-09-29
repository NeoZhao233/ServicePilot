# Failure Memory 开发与验证指南

## 启用

仅动态 LangGraph 路径接入。设置 `SERVICEPILOT_DYNAMIC_PLANNING=1` 和 `SERVICEPILOT_FAILURE_MEMORY=1`。默认保持关闭以便 B0 基线和灰度比较。`SERVICEPILOT_FAILURE_MEMORY_DB` 可指定独立 SQLite 文件；未设置时使用业务 SQLite 路径，但经验仍在独立 `failure_*` 表中。若 CLI 传入 `chroma_dir`，使用其中独立的 `failure_memory` 集合；SQLite 始终是唯一真实来源。`SERVICEPILOT_FAILURE_MEMORY_MODEL` 可选择 DeepSeek 结构化抽取模型；在线模式未设置时沿用已配置的意图模型。离线模式使用受限确定性模板。

实验时也可以用 `SERVICEPILOT_FAILURE_MEMORY_MODE=B0|B1|B2` 显式选择组别，优先于简易开关。B0 不接入记忆；B1 把所有观察到的失败反思存入实验表，并在每次候选规划时全量注入；B2 使用经过验证的结构化记录和 Top-K 检索。B1 仅用于对照，反思可能包含非 Agent 故障。

```sh
SERVICEPILOT_DYNAMIC_PLANNING=1 SERVICEPILOT_FAILURE_MEMORY=1 \
  .venv/bin/python -m servicepilot.cli --data-dir data --session-db .servicepilot/state.sqlite
```

`FailureMemoryStore` 支持 `add_or_merge`、`get`、`list_active`、`invalidate`、`rebuild_index` 和 `record_usage`；`FailureMemoryRetriever.retrieve` 限制最多 5 条，默认 3 条。`FailureAnalyzer.analyze` 只有在独立验证器、业务规则或测试 oracle 提供失败断言，或持久账本证明新操作重复提交时才给 `worth_storing=True`。动态运行时会为计划校验失败、PolicyGuard 拦截和旧计划执行尝试创建候选；RAG 错判仍需要外部独立验证结果，不会因工具报错自动入库。

规划前的召回以 `verified_failure_lessons` 传给候选规划模型，最终计划仍须通过 `TaskPlanner.validate`。工具前与恢复时仅记录相关召回和理由，不让记忆执行工具。`FAILURE_MEMORY_APPLIED` 只在经验确实进入 Planner 输入后写入。Trace 中保存 memory ID、score、阶段和适用理由。SQLite `failure_memory_usage` 保存使用状态；动态任务完成并通过账本核验时记录成功结果。其他任务结果需由独立业务 oracle 补写，不能把暂时的 `waiting_confirmation` 判成失败。

写入采用候选事件和终态核验：受限写入被 PolicyGuard 拦截、旧计划进入执行器、经过外部 oracle 判错的 RAG 判断，以及工具异常后使用新 operation ID 重复提交，先保存到任务状态并记 `FAILURE_CANDIDATE_CREATED`。任务完成、安全暂停或转人工时，再用 Trace、计划版本和业务账本核验；合格时记录 `FAILURE_VERIFIED` 并调用原有经验提取器。普通超时、正常重规划和正确安全降级不生成可召回经验。根因无法确认的高风险候选留在 `failure_candidates` 表待人工审阅，不进入向量索引。RAG 判断须由外部独立验证器调用 `queue_independent_rag_failure` 提供规则与证据引用；当前运行时不会从检索为空推断 Agent 判断错误。

普通经验容量由 `SERVICEPILOT_FAILURE_MEMORY_CAPACITY` 设置，默认 100；重复退款、越权及绕过确认类经验自动 pinned，不占普通容量。去重以任务类型、失败类别、工具和已验证根因为准，合并增加次数与 Trace 引用，并保留原 lesson。仅经验进入 Planner 输入时更新 `last_accessed_at`；检索命中不更新。超限时按最后实际注入时间淘汰，未注入过的按创建时间。淘汰时删除 SQLite 与 Chroma 条目；向量删除失败则立即尝试重建，启动时也会再同步。召回只读取 SQLite 存在的记录。现有 `FAILURE_MEMORY_APPLIED` 即注入事件，避免另记同义事件。

在线意图分类可选设置 `SERVICEPILOT_FAILURE_MEMORY_ROUTING=1`，只把已验证且 `task_type=intent_routing` 的经验注入 `ResponsesIntentModel`；默认关闭。此类错误没有可靠的自动真值，需人工或独立 oracle 核验后通过现有存储入口写入。冻结合成挑战集上的[真实模型意图实验](../../evaluations/FAILURE_MEMORY_INTENT_REPORT.md)和[同类错误复发率实验](../../evaluations/FAILURE_RECURRENCE_REPORT.md)显示该切片有正向结果，不能与默认 Planner 的指标混算。

## 经验污染案例

| 轨迹 | 错误做法 | 当前防护 |
| --- | --- | --- |
| 退款调用超时，账本显示已成功 | 记为工具坏了并建议重试 | 超时不能证明 Agent 责任；既有恢复策略查询原 operation ID。 |
| 旧版政策曾要求转人工，新版允许自动流程 | 把旧经验继续套到新任务 | 记录可绑定政策版本；失效后 `active` 过滤，变更须调用 `invalidate`。 |
| 同一退款 operation ID 再次查询返回去重 | 误判重复副作用 | 只有账本证明新操作再次提交时才归类 `DUPLICATE_SIDE_EFFECT`。 |
| 检索为空，Agent 安全转人工 | 记为检索失败经验 | 无独立 oracle 证明漏检且给出错误确定答复时不入库。 |
| 无关换货经验匹配退款任务 | 全量注入 | task_type、工具、适用条件和版本过滤；Top-K 限额。 |
| 两条经验互相冲突 | 把经验当作业务规则 | 记忆只是 Planner 上下文；现有业务规则、授权和验证器仍决定动作。冲突样本需在评测中显式标注。 |

## B0/B1/B2 对照实验

先固定订单、政策、工具契约、模型版本、随机种子和任务顺序；为每种失败模式构建首次失败和后续不同订单任务。三组的启用方式如上。B1 实验注入器是评测基线，不应在生产路径启用。六类情景、指标公式和公开 Benchmark 边界见 [技术设计](failure_memory_design.md)。

每次任务由独立业务断言生成 JSONL `EvaluationRun`，包含 `oracle_ref`、`seed`、`model_id`、`policy_version` 和 `tool_version`；三组 `case_id` 及这些控制条件必须一一对应。运行：

```sh
.venv/bin/python -m servicepilot.failure_memory_evaluation runs.jsonl --output report.json
```

评测入口校验分组、配对案例和计数，输出重复失败率、任务成功率、首败后成功率、错误召回/使用率及开销。它只汇总已经执行并由独立 oracle 标注的任务，不会生成模型运行结果。另有[固定离线演示](../../evaluations/run_failure_memory_showcase.py)及[逐例结果](../../evaluations/results/failure_memory_showcase.json)，使用脚本化规划模型展示 B0/B2 链路。[真实模型配对测试](../../evaluations/FAILURE_MEMORY_REAL_REPORT.md)在 24 组虚构任务上观察到 B0/B2 均为 0 次计划校验错误，未证明真实模型的改善。当前仍没有带标注 AgentErrorBench 运行或足够的真实失败后续任务，因此不提供性能提升数值。[AgentDebug 项目](https://github.com/ulab-uiuc/AgentDebug)公开了 AgentErrorBench 的标注轨迹下载入口；公开 Benchmark 应用独立 adapter 和独立报告，不能与售后场景混算。

## 验证

```sh
.venv/bin/ruff check src/servicepilot/failure_memory.py src/servicepilot/failure_memory_evaluation.py tests/test_failure_memory.py tests/test_failure_memory_evaluation.py
.venv/bin/pytest -q tests/test_failure_memory.py tests/test_failure_memory_evaluation.py
```
