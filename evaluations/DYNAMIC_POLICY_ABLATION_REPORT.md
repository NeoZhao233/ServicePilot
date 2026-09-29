# 动态执行路径与静态路径的配对任务实验

## 结果

在同一个 SQLite 安全恢复底座上，仅切换 `dynamic` 开关，运行 12 道事先写定的虚构售后任务。6 道任务模拟政策检索返回空，6 道任务有正常政策证据；退款任务包含用户确认回合。按最终回复和业务账本判分：

| 场景 | 静态路径 | 动态路径：模板计划 | 动态路径：DeepSeek 候选计划 |
| --- | ---: | ---: | ---: |
| 政策检索为空，6 道 | 0/6 | 6/6 | 6/6 |
| 政策正常返回，6 道 | 6/6 | 6/6 | 6/6 |
| 合计 | 6/12 | 12/12 | 12/12 |

题集 SHA-256：`dc3fd2dfa4c541c02103f85bce95427f1136cdaa83b51bfb1bd3af82c8e3fae6`。逐题回复、政策命中数和退款、工单、人工队列记录见[模板计划原始结果](results/dynamic_policy_ablation.json)和[真实模型原始结果](results/dynamic_policy_ablation_deepseek.json)，固定题目见[案例集](dynamic_policy_ablation_cases.json)。真实模型组使用 `deepseek-chat` 生成候选计划，共 12 次规划、6,018 Token，没有计划结构校验错误；其 6 条失败事件均为预期注入的政策证据缺失。

“成功”由外部断言定义：政策证据齐全时，退款必须请求确认并在确认后恰好创建一笔退款，换货必须恰好创建一张工单；没有政策证据时，必须暂停并转人工核对，账本不能新增退款或工单。评分不读取 Agent 对自身的成功判断。

## 观察到的机制

静态路径在政策检索为空时，仍提交了 3 笔退款并创建了 3 张换货工单。动态路径在计划中的 `search_policy` 步骤发现没有命中后停止，6 道任务均无业务写入且进入人工队列。政策正常时两组均完成 6/6，说明动态路径没有因新增前置步骤而阻断这些任务。

这个结果支持**当前动态规划与执行路径中，显式政策证据前置步骤提高了异常场景的任务结果正确率**。它不证明“目标改变后的重新规划”单独提高了成功率：这 12 道任务没有目标变化，每个动态任务只规划一次。在此前 9 类、每类重复 3 次的[隔离对照实验](../docs/UPGRADE_DELIVERY.md)里，完整动态路径与同安全底座的静态路径都是 27/27。若将相同政策证据门槛加到静态路径，本实验差距也可能消失。

## 复现与边界

```bash
PYTHONPATH=src .venv/bin/python evaluations/run_dynamic_policy_ablation.py
PYTHONPATH=src:evaluations .venv/bin/python evaluations/run_dynamic_policy_ablation.py \
  --planner-online --allow-online --model deepseek-chat \
  --output evaluations/results/dynamic_policy_ablation_deepseek.json
```

这是本地业务故障注入实验：政策检索为空由测试固定注入，订单与政策均为虚构数据。第一轮使用项目默认的确定性三步模板；第二轮只在动态组调用真实模型生成受校验的候选计划。两轮使用**同一题集**，不能视为 24 道独立任务。12 道题中缺失证据任务占 50%，这只是实验配比，不能解释成真实流量中的总体成功率或线上提升。没有测量动态重规划的独立收益、真实支付渠道或生产延迟。
