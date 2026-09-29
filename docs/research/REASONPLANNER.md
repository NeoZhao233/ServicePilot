# ReasonPlanner 适配

材料：[arXiv v1 HTML，第 3–4 节与 Algorithm 1](https://arxiv.org/html/2410.09252v1)。已核对方法与 ScienceWorld 实验；未运行官方实现或复现 ScienceWorld。

论文贡献：以时序知识图谱构建世界模型，生成假设轨迹，再以 actor/critic 比较实际执行与预测结果来修订计划。实验反映模拟科学任务中的表现，不能作为电商事务一致性保证。

现有对应：原系统每轮查询订单，但没有可持久化的步骤依赖和计划版本。

工程适配：planning.py 的 TaskState/TaskPlan/PlanStep 保存目标版本、计划版本、观察快照与步骤状态。StateReconciler 取消失效的未执行步骤、保存旧计划历史。当前轮的订单读取可以复用，敏感提交仍必须原子复核最新事实。

自主设计：SQLite 订单 version、整数分金额、候选计划 Schema、DAG 校验、跨轮授权失效、调用预算以及 completed/pending 派生属性。它们不是论文实现的复制。

未采用：图数据库、LLM 世界模型预测、完整 actor/critic 双模型。首期只允许退款申请与换货受理的三步骤计划；真实库存和换货履约工具尚不存在，因此不编造那些步骤。

对应验证：test_model_cannot_bypass_execution_contract、test_valid_non_list_order_is_scheduled_by_dependencies、test_amount_change_invalidates_confirmation、test_state_change_at_write_is_rejected。模板是默认强基线，ChatCompletionPlannerModel 仅生成候选 JSON。
