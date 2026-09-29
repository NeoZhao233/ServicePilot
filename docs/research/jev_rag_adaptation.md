# Jev / CRAG 研究与 ServicePilot 适配边界

后续实现见 [Jev 自纠错 RAG](jev_corrective_rag.md)：已安装并核验 jev-reranker 0.1.2，以下保留阶段一研究记录。

核查日期：2026-09-24。本轮交付阶段一审计与原始基线；未替换生产检索器、未安装 Jev 依赖、未启用过滤或纠错。以下接口以读取的官方资料和上游源码为依据，尚非本项目安装验证结果。

## 资料身份与可核验接口

官方域名由 [TypeSafe 官网](https://typesafe.ai/) 及其链接的 [Python SDK 文档](https://docs.typesafe.ai/sdk/python/) 核验；不采用名称相似的聚合站作 API 依据。

[官方 SDK](https://github.com/typesafe-ai/typesafe-sdk-python) 的发行包名是 `typesafe-sdk`，导入名是 `typesafe_sdk`。读取的 main 分支 [pyproject.toml](https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-python/main/pyproject.toml) 声明 0.7.1、Python >=3.10，依赖 httpx2、Pydantic、tenacity；这不是本地已安装版本，也不是固定 commit 承诺。本地 Python 3.11 满足语言要求，但仍需后续隔离解析完整依赖。

官方同步 `TypeSafeClient` 与异步 `AsyncTypeSafeClient` 支持 `system_one(state=..., questions=...)`。输入是文本或 JSON 状态、命名的 Choice/Score/Noul 问题。Choice 返回选择、概率分布和 confidence；Score 返回评分、分布和 confidence；Noul 是真假概率值。结构合法不保证判断正确。[官方入门](https://docs.typesafe.ai/introduction)、[返回类型](https://docs.typesafe.ai/sdk/python/api/types/responses)、[置信度说明](https://docs.typesafe.ai/confidence)。

[异步接口](https://docs.typesafe.ai/sdk/python/api/clients/async) 可在客户端或单次调用指定 timeout、retry；`RetryPolicy(max_retries=0)` 禁用重试。默认响应包含模型和 usage。连接、API、响应校验错误分别处理；实际费用未确认存在统一返回字段，应保留未知。官方 [Usage](https://docs.typesafe.ai/sdk/python/usage) 提醒 debug 会记录请求响应正文，ServicePilot 不应开启未脱敏的 SDK debug。

## 第三方 jev-reranker

[项目文档](https://github.com/hotchpotch/jev-reranker) 与 [源码](https://raw.githubusercontent.com/hotchpotch/jev-reranker/main/src/jev_reranker/reranker.py) 属于 hotchpotch，不是官方 SDK。本轮读取 main 的 [包声明](https://raw.githubusercontent.com/hotchpotch/jev-reranker/main/pyproject.toml) 为 0.1.2、Python >=3.11，基础依赖 httpx 和 python-dotenv。本地未安装，后续必须锁定发行版并验证签名，不把 main 文档等同于发布包。

已核实接口：`rerank(query, documents, *, instruction=None, threshold=0.0, top_k=None, return_documents=True, detail=False)`；`relevance_rerank` 对应默认 threshold=0.2。异步分别为 `a_rerank`、`a_relevance_rerank`。两者区别包括评分指令，不能仅视为不同阈值。返回 `results`，以 `document_index` 对齐原始候选；`detail` 提供执行和被排除文档信息。

支持 listwise、pointwise；relevance 模式不支持 pairwise。`instruction` 是 instructions/criteria 字典，非任意字符串。listwise 分块逐次处理；实例共享并发限制，默认 listwise 4，pointwise/pairwise 20。默认单次 HTTP timeout=180 秒、最多 8 次重试，不宜直接用于客服请求。`top_k` 不减少评分工作。detail 中的输入输出不能整包写入 ServicePilot Trace。

[HTTP 实现](https://raw.githubusercontent.com/hotchpotch/jev-reranker/main/src/jev_reranker/_client.py) 校验每个 Noul 分数、全部问题键、模型和非负 usage；传输失败及指定限流/服务端状态有界重试，取消会传播。响应 usage 缺失时会报校验错误，不能当作实际零用量。

## CRAG 原始机制与自主设计的区分

[论文](https://arxiv.org/abs/2401.15884)、[全文](https://arxiv.org/html/2401.15884v3)、[作者代码](https://github.com/HuskyInSalt/CRAG) 的原始机制是：训练的轻量检索评估器给出相关性信号，形成 Correct/Incorrect/Ambiguous 动作；提炼知识片段，并按情况加入网页检索。其知识提炼包括分解、过滤和重组。论文实验成绩不能迁移为 ServicePilot 的收益。

ServicePilot 拟采用的四态充分性（SUFFICIENT/INSUFFICIENT/CONFLICTING/UNCERTAIN）、订单条件校验、官方政策来源及有效期检查、最多两轮定向检索、无新证据停止和人工降级，均是本项目设计，既不是 CRAG 原样复现，也不是 Jev 或第三方库提供的自动售后保障。Jev 只提供候选结构化判断能力，是否足以判断中文复杂售后证据，需要独立数据验证。

## 后续实施门槛

1. 保留原始检索参数、冻结样本及本轮候选快照，先补齐可运行 B0 在线基线。在线结果不得用哈希向量或 Mock 冒充。
2. 通过现有 PolicyReranker 接口增加可选 Jev 适配。索引先验证唯一性、数量和范围，再映射完整 RetrievalHit；异常使用原候选降级。
3. 先运行不删文档的排序和影子评分，再在独立校准集确定阈值，冻结测试集只用于最终比较。第三方默认 0.2 不能作为上线阈值。
4. 独立评估充分性，尤其关注“高相关但缺关键例外条款”。显式日期、类别、版本条件由确定性规则处理。缺字段应判未知，不创造元数据。
5. 原始问题和业务上下文目前没有完整传入检索接口，需明确最小 RAG 接口扩展；不改规划决策或工具安全。定向重写由缺失维度映射受控查询，不假定 Jev 会自由生成查询文本。
6. B0–B3 使用同一初始候选、同一标注和统一答案评判。分别记录真实请求/重试/Token、成本未知项及端到端耗时；B3 额外召回单列。线上试验设请求上限与失败熔断。
7. 不把普通网页当作有执行效力的售后政策；只有审核过的政策库可提供业务依据。

原始在线基线补测记录见 [在线基线补充](jev_stage1_online_followup.md)。

本轮尚未做 Jev 评分分布、充分性可靠性、过滤阈值校准或 B1–B3 消融，因此没有声称任何 Jev 准确率、速度或成本收益。
