# 线上摘要实验的外发数据检查

本实验的 `--live-summary` 路径只构造新的内存 `ServiceAgent` 与 `InMemoryCommerceRepository`，每例独立创建；不读取现有会话数据库、用户上传文件、Trace 数据库或 Failure Diary。订单来自仓库的 `data/orders.json` 演示种子（`USER-001`、`ORD-1001` 等虚构 ID），政策来自 `data/policies.json` 演示语料；案例问题和跟进话术都在 `evaluations/run_context_compaction_eval.py` 的 `CASES` 常量及循环中直接生成。

外发请求目标由 `.env` 的 `SERVICEPILOT_BASE_URL` 指定，当前为 DeepSeek API。预检只发送“售后”一词。实验摘要请求只发送旧 `ContextSummary` 和经 `ContextCompactor._select` 筛选的旧对话消息。对话内容可能包含上述虚构订单 ID、模拟订单状态、模拟退款金额、模拟工具回复及合成跟进话术。API key 用作请求认证，不写入实验结果。结果文件保存每例模型 usage、延迟及模拟任务指标。

这项检查证明实验外发的是仓库自带的虚构演示数据，而非已有用户会话；它不替代用户对向 DeepSeek 发送该数据的授权。
