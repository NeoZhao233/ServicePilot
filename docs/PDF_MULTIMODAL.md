# ServicePilot 用户 PDF 问答

## 范围与现状

改造前只有 JSON 聊天接口，没有 PDF 上传路径。已有 BM25、BGE/向量、RRF、Evidence Evaluator、LangGraph 主图和 Trace；这些组件没有重写。用户上传的 PDF 只进入用户与会话隔离的文档库，不写入企业官方政策数据或索引。PDF 回答提供用户文件中的证据；涉及企业政策和实际售后操作时，原有 Agent 与 Evidence Evaluator 仍按原路径工作。

当前覆盖可提取文字、表格、嵌入图片和部分扫描件的日常 PDF。它不是任意 PDF 的准确率保证：复杂跨页表格、手写内容、低质扫描、图片里的细节仍可能失败。

## 最小改造

- `pdf_fusion.py` 用单一 Docling PDF pipeline 提取文字、Markdown 表格和图片，开启其 OCR 与表格结构选项。每个元素保存 `document_id`、页码、元素 ID、类型、内容、图片路径、关联元素 ID，并保存 Docling 的类型标签和位置框。跳过页眉页脚、过小的图片、明确标注为 Logo 或装饰的图片，以及无图注且位于页角的装饰图。图片区域内的 OCR 片段不作为独立正文；用原生图注引用和同页几何位置建立有限关联。关联只表示共同出现，不证明图注描述就是图片中的事实。
- `MultiModalFuser` 把长表格拆成带表头的行证据，文字直接索引，图片索引图注和关联说明并保留原图。冒号后的相邻值行保持关联。检索复用现有 BM25、向量和 RRF，另对短字段、表单行号和图例编号做小规模词项重排。返回证据只引用户 PDF 的文档 ID 和页码。
- “分别、比较、列出”等问题可返回最多 3 条不同页码优先的原文证据。视觉问题指定图号时按图注选择对应图片；检索到多张图片且无法唯一确定目标时，停止视觉调用并提示证据不足。
- 可选的 `SERVICEPILOT_PDF_ANSWER_MODEL` 复用已有 Responses 传输层，对检索到的最多 4 条文字/表格证据做归纳。模型须返回逐字支持引文和证据 ID；系统核对引文、数字与证据页码，核对失败或模型不可用时返回原文。图片问题不会交给这个文本模型。
- `pdf_service.py` 将原 PDF、图片和元素元数据保存到 `SERVICEPILOT_PDF_DIR`，默认位于会话数据库旁的 `pdfs/`；按用户 ID 与会话 ID 检查读取、列举、图片预览和删除。若未配置持久会话数据库，`create_app` 使用临时目录。上传可自动建立仅供文档问答的会话，限制为 15 MB。
- `webapp.py` 提供 `/api/pdf/upload`、`/api/pdf/documents`、`/api/pdf/ask`、受保护的图片预览与删除接口；现有 `/api/chat` 可带 `document_id` 提问。网页可上传、选择、预览和删除 PDF。
- 只有问题确实需要图像观察时，才尝试 `SERVICEPILOT_VISION_MODEL`。适配器向明确支持图像的 Responses 服务发送 PNG 输入。没有模型、图片未成功解析或调用失败时，明确返回图片证据不足；不会把文本模型当作视觉模型。
- 现有 Trace 记录解析开始/结束、元素数量、耗时、检索文档和页码、视觉调用状态、耗时与服务返回的 token 用量。

## 演示与解析示例

```bash
uv sync --extra agent --extra api --extra semantic --extra pdf --extra dev
uv run servicepilot-web --port 8765
```

打开网页，上传 `evaluations/pdf_fixtures/inspection.pdf`，再问“文字检测结论是什么？”。返回第 3 页的“设备主板损坏”文字证据。问“照片里是否能看到主板烧毁？”时，若未配置视觉模型，会明确提示图片证据不足，并提供可预览的原图引用。

| 类型 | PDF 与页码 | 提取结果 |
|---|---|---|
| 文字 | `policy.pdf` 第 1 页 | `质量故障不受未拆封条件限制，需提供检测材料。` |
| 表格 | `conditions.pdf` 第 1 页 | `| 质量故障 | 保修期内 | 检测后维修或换货 |`，同表保留表头 |
| 图片 | `inspection.pdf` 第 3 页 | 主板检测图片的 PNG 引用，关联图注与同页检测结论；不自行断定图片可见烧毁 |
| 扫描文字 | `scan_receipt.pdf` 第 1 页 | `Device SP-200 Repair date: 2026-09-20 Warranty: 12 months` |

`invoice.pdf` 的 12 行明细检验行级检索，`manual.pdf` 检验跨两页说明和图 2，`gallery.pdf` 检验并排的两张图，`decorated.pdf` 检验页脚装饰过滤。这些 PDF 由 `evaluations/build_pdf_fixtures.py` 生成，仅用于可复现评测；不是用户提供的真实业务 PDF。

API 上传示例：

```bash
curl -F user_id=USER-001 -F file=@evaluations/pdf_fixtures/inspection.pdf \
  http://127.0.0.1:8765/api/pdf/upload
```

响应含 `session_id`、`document_id` 和分类型数量。随后向 `/api/pdf/ask` 发送这两个 ID、`user_id` 与 `question`。若启用了鉴权，同时提供与 `user_id` 匹配的 Bearer token。只有实际可用的图像服务才应配置 `SERVICEPILOT_VISION_MODEL`。若需要归纳式文字回答，可另配 `SERVICEPILOT_PDF_ANSWER_MODEL`；两者都使用 `SERVICEPILOT_API_KEY` 和 `SERVICEPILOT_BASE_URL`。配置文本回答模型意味着检索出的用户 PDF 片段会发给该模型服务。

## 同题对照

运行 `.venv/bin/python evaluations/run_pdf_eval.py`。20 个固定问题和预期短语见 `evaluations/pdf_cases.json`；逐题输出及计时见 `evaluations/results/pdf_eval.json`。B0 是对同一 Docling 解析结果做纯文本展平、丢弃图片的模拟基线，因为旧系统没有 PDF 提取路径。B1 是分类型证据与相同的 BM25+BGE+RRF。两组使用同样的问题、PDF、OCR 结果和本地 BGE；没有调用视觉模型。

| 指标 | B0 | B1 |
|---|---:|---:|
| 回答包含预期短语或正确拒答 | 17/20 | 20/20 |
| 检索候选包含正确页码 | 20/20 | 20/20 |
| 6 条表格题中保留 Markdown 行列 | 0/6 | 6/6 |
| 4 条相关图片题中保留原图引用 | 0/4 | 4/4 |
| 视觉模型调用及 token | 0 | 0 |
| 20 题检索与证据返回总耗时 | 20.88 ms | 23.42 ms |

这不是生成答案的语义正确率。短语命中和候选页码是有限指标；“正确页码”也不代表首条证据必然正确。解析 8 份合成 PDF 共约 8.17 秒，B0/B1 建索引共约 23.36/62.17 ms。耗时来自一次本机暖缓存运行，不能当作稳定性能结论。

另外用两份公开、非合成 PDF 做小规模审查：[IRS W-9（6 页）](https://www.irs.gov/pub/irs-pdf/fw9.pdf) 与 [CFPB 信用卡账单样本（1 页）](https://files.consumerfinance.gov/f/documents/cfpb_building_block_activities_sample-credit-card-statement_handout.pdf)。运行 `python evaluations/run_public_pdf_audit.py --w9 /path/fw9.pdf --statement /path/statement.pdf`，结果在 `evaluations/results/public_pdf_audit.json`。W-9 提取 225 段文字、4 张表；账单提取 12 段文字、4 张表。5 个字段问题均在首条带页码证据中包含预期值。两份 PDF 解析并建索引分别约 17.81 秒和 2.77 秒。这只验证了表单和账单的几个字段，不代表一般日常 PDF 已全面通过。

第三份公开样本是 [HP Deskjet 845c/825c 快速参考手册（22 页）](https://www.hp.com/ctg/Manual/bpd09054.pdf)，含两栏文字、三列表格、部件图和页边印刷装饰。运行 `python evaluations/run_manual_pdf_audit.py --manual /path/bpd09054.pdf`，逐题结果在 `evaluations/results/manual_pdf_audit.json`。解析出 304 段文字、7 张表、2 张有用示意图；页边装饰图被过滤。6 个问题均返回含预期值且页码正确的证据，其中图片观察题因未配置视觉模型正确拒答，并保留原图引用。解析与建索引约 33.66 秒。这个样本比合成 PDF 更复杂，但仍只是一本手册，不能代表所有日常排版。

## 已知边界

- 没有配置真实视觉服务，因此图片事实识别和实际视觉 token/延迟尚未测量；单元测试只验证接口确实发送图片字节并读取服务用量。
- 图片过滤是保守启发式，未覆盖所有 Logo、水印和装饰；复杂多栏排版或跨页表格仍需人工核验。
- 未配置回答模型时，跨页问题列出多条引用式原文；配置后可尝试归纳，但引文核对不能证明所有语义推断都正确，且不会执行无法核验的跨页计算。用户应检查页码和原文，尤其是金额、日期和表单字段。当前没有真实在线回答模型的质量与 token 实测，相关单元测试使用模拟服务。
- 文档保存在服务器本地，用户可从页面删除；部署时应为该目录设置备份、保留期限和访问控制。
