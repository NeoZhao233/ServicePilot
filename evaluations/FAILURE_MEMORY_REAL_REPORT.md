# Failure Memory 真实模型配对测试

## 结论

本轮没有观察到失败日记让真实模型减少规划错误。24 组配对任务中，B0 和 B2 的计划校验错误均为 **0/24**。B2 确实在 24 次请求中注入了经验，但基线已经全部通过，因而无法从这组数据证明增益；也不能据此断言失败日记在其他任务上无效。

## 方法

- 模型：`deepseek-chat`，`temperature=0`，同一模型和同一 Planner 模板。
- 任务：12 个冻结的、完全虚构的规划挑战，每题重复 2 次；每次配对任务只改变是否注入失败经验，组别调用顺序交替。
- B2 经验：先用现有 `TaskPlanner.validate` 独立证实一个虚构的错误计划，再由现有经验提取器写入一条 `PLANNING_ERROR`。它是**合成先例，不是 DeepSeek 自己犯过的错**。
- 评分：本地 `TaskPlanner.validate` 检查步骤、工具、订单参数、依赖和授权前置条件。模型服务异常不会算作计划错误。
- 数据：只向外部模型发送 `SYNTHETIC-*` 标识和虚构测试文本，不发送项目订单或用户数据。

| 指标 | B0：无失败日记 | B2：注入失败日记 |
| --- | ---: | ---: |
| 计划校验错误 | 0/24 | 0/24 |
| 实际经验注入 | 0 次 | 24 次 |
| 输入与输出 Token 合计 | 12,382 | 13,010 |

逐题结果见 [原始 JSON](results/failure_memory_real_eval.json)，冻结任务见[案例集](failure_memory_real_cases.json)。

## 后续探索：寻找真实模型首次失败

为避免只在一个模型上得出结论，又用 `deepseek-flash` 对同一批 12 个合成挑战做了**仅 B0 的探索性训练轮**。结果仍是 **0/12 次计划校验错误**，见[探索记录](results/failure_memory_flash_discovery.json)。这轮没有产生可由真实模型错误提取的经验，因此没有把这些题再包装成“失败后学习”的测试，也没有继续通过修改题目追求正向数字。

另做了一次探索性探测：不给 `deepseek-flash` 完整 JSON 模板，只提供同等的步骤与安全约束文字，模型仍通过本地计划校验。该单次探测未进入正式指标，也不能用于声称效果。

## 复现

在本地 `.env` 配置可用的 `SERVICEPILOT_API_KEY`、`SERVICEPILOT_BASE_URL` 和 `SERVICEPILOT_INTENT_MODEL` 后运行：

```bash
.venv/bin/python evaluations/run_real_failure_memory_eval.py \
  --allow-online --model deepseek-chat --repeats 2 \
  --output evaluations/results/failure_memory_real_eval.json
```

该命令会向外部模型发送 48 次合成任务请求，产生 API 费用。不同时间的模型服务可能返回不同结果；报告保留了本次逐题结果与 Token 用量。

## 解释边界

这次是 Planner 层测试，验证的是模型在不可信备注下能否维持受限计划结构；它没有测量完整客服对话的成功率，也没有测量长期学习。模板和校验器本身足够强，导致本组任务没有暴露可供记忆减少的错误。若要声称“真实模型因失败日记更少犯错”，需要先收集真实模型的已验证失败模式，再用未参与经验提取的后续任务做配对评估，并报告失败率、业务完成率和额外成本。
