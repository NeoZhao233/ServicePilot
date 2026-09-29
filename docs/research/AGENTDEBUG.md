# AgentDebug 适配

材料：[论文第 2–4 节与 Algorithm 1](https://arxiv.org/html/2509.25370v1)、[官方核心诊断源码](https://raw.githubusercontent.com/ulab-uiuc/AgentDebug/main/detector/critical_error_detection.py)。阅读了 CriticalErrorAnalyzer.identify_critical_error 的提示构建、解析和有界重新分析；未运行论文环境。

论文贡献：按模块和步骤分类轨迹错误，定位导致失败的关键原因，以针对性反馈改善重新执行。源码的诊断重试是模型分析重试，不能等价为业务写接口重试。

现有对应：ToolExecutor 原本对错误结果、连接失败和超时使用统一重试，不能区分响应丢失与未提交。

工程适配：recovery.FailureAnalyzer 根据结构化错误码选择 repair_plan/reconcile/reconfirm/query_operation/escalate；RecoveryPolicy 查询原 operation_id。成功但丢响应是查询后的诊断结论，不是超时发生时可以凭空得知的事实。

自主设计：prepared/succeeded/unknown 操作账本、冻结、持久化人工队列、一次查询后的保守处理，以及退款写入禁止自动重试。参数非法候选计划只回退一次确定性模板，不循环让模型修复。

未采用：从关键错误步骤重置整个现实业务环境。退款提交不能撤销重放；LLM 也不能判定金融操作是否成功。

对应验证：test_response_loss_queries_without_second_write、test_unknown_freezes_and_survives_restart、test_query_failure_does_not_cause_resubmission、test_malformed_model_falls_back_once_without_authorizing。错误原因和原结果保存在 task.failure_events 与人工处理队列。
