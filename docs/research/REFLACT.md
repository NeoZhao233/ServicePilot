# ReflAct 适配

材料：[arXiv v1 HTML，第 4、6 节及表 2](https://arxiv.org/html/2505.15182v1)。已阅读核心方法与实验，不声称复现其基准。

论文贡献：让行动选择持续基于当前观察、内部状态与任务目标的关系，减少脱离环境的推理。实验在 ALFWorld、ScienceWorld、Jericho；这不是退款服务可靠性实验。

现有对应：原有 evaluate_refund 检查资格，graph_runtime 检查回复与工具事件是否一致；没有任务级目标完成状态。

工程适配：planning.GoalStateEvaluator 基于结构化步骤，dynamic_runtime._evaluate 查询退款/工单账本验证实际效果；StateReconciler 用目标和订单快照的差异驱动失效处理。模板或模型都不能把自然语言自评当作业务事实。

自主设计：金额授权、事务、operation_id、unknown 冻结、跨进程会话锁都属于项目可靠执行协议，不属于 ReflAct 算法。

未采用：每一步让 LLM 生成内部信念与反思。当前退款状态可从权威系统获得，增加模型判断反而弱化确定性与可测性。

对应验证：test_success_shaped_response_is_not_completion、test_task_history_reuses_verified_read、test_switch_after_success_does_not_undo_refund。原始结果见 dynamic_ablation.json。没有测得在线模型规划的增量收益，不迁移论文提升比例。
