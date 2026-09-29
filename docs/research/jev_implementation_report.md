# Jev RAG 本轮实施报告

本轮完成实验实现，默认仍为 original；未向 Jev 发送任何数据，未修改商务工具、任务规划算法或记忆模块。

| 阶段 | 修改 | 验证与边界 |
| --- | --- | --- |
| Jev 评分 | jev.py、可选依赖与 uv.lock | 实装 0.1.2 异步接口；真实库经 Mock HTTP 验证映射、用量、超时、取消、无重试和全局 HTTP 并发限制。没有实网评分 |
| 证据判断 | evidence.py、evidence_calibration.py、PolicyChunk 可选元数据 | 四态输出、确定性条件/日期/来源/版本检查；校准与独立验证 gate 单测通过，尚无真实放行 artifact |
| 有界纠错 | corrective_rag.py | 定向查询、最多两次纠错、无新有效证据停止、只对最终 Top K 作判断；未知/冲突不释放证据 |
| 接入与 Trace | cli.py、mcp_server.py、Agent/Graph 装饰器、统一节点包装、观测事件及既有查看页事件过滤 | 默认路径保持；原始问题与必要条件传递，实际 HTTP 调用独立计量；动态路径空证据进入原有人工核对。旧的不支持空证据降级的分支保留 original |
| 评测准备 | rag_evaluation.py、六条补充样本、消融状态 JSON、原 B0 源码快照 | 原 24 题及原始基线文件未改；B0 实测保留，B1–B3 not_run，答案/费用等未测项保持 null |
| 回归 | test_corrective_rag.py、基线脚本测试的独立 fixture | 新增 30 项测试，全量 159 项通过；Ruff 通过。1 条已有依赖弃用警告 |

补充说明：为了防止升级后的代码被当作原始 B0，本轮保存了四个与阶段一 manifest 的 SHA256 完全一致的源文件快照。基线脚本的预算单测使用独立临时 fixture，未篡改真实基线 fingerprint 或放松生产脚本的源代码校验。

后续未完成事项：配置本地 TYPESAFE_API_KEY；在单独授权的预算内采集真实 Jev 分数；建立与冻结集分离的校准/充分性验证集；补充经过核实的政策来源、生效日期和业务版本；执行 B1–B3 及答案层人工或统一 judge 评测。此前不能启用证据过滤/纠错，也不能认定 Jev 比原实现更快、更省或更准确。

完整配置、接口、工作流和限制见 [使用说明](jev_corrective_rag.md)，原始实测见 [B0 在线报告](jev_stage1_online_followup.md)。
