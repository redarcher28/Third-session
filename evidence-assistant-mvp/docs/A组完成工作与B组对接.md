# A 组完成工作与 B 组对接

> 更新日期：2026-08-11
> 适用仓库：`evidence-assistant-mvp`

## 1. 当前交付状态

A 组的数据采集、标准化、追溯、切分、Wiki 和隔离向量库建库链路已具备。2026-08-11 已完成一次不访问在线 API 的隔离建库验证：

- 本地 EvidenceDoc：7 条；
- Wiki 文档：8 条；
- 合并后的文档：15 条；
- 隔离 Chroma：15 个 Chunk；
- `documents.json`、`wiki_docs.json`、`documents_with_wiki.json` 中每条记录均含 `citation_eligible`、`record_type`、`source_locator`、`provenance`。

本次验证使用的向量库目录为 `data/chroma-a-validation/`，没有修改默认的 `data/chroma/`。

## 2. A 组实现内容

### 2.1 统一数据与引用边界

`EvidenceDoc` 和 `Chunk` 共享以下可追溯字段：

- `citation_eligible`：是否允许作为对外引用；
- `record_type`：记录类别；
- `source_locator`：稳定来源定位符；
- `provenance`：采集 run、原始响应路径、查询词、去重血缘等信息。

`trial_registration` 固定为 `evidence_level="other"`，只可引用 NCT、状态、干预、终点、结果登记等事实，不能作为疗效证据。Wiki 和教学样本不可作为临床证据引用。

### 2.2 多源采集

| 来源 | 入口 | 原始响应 | 标准化与追溯 |
| --- | --- | --- | --- |
| PubMed | `ingest_pubmed()` | ESearch JSON、EFetch XML | 保留 PMID、DOI、文章类型、全部命中查询；每篇文献仅关联实际承载它的 EFetch XML 批次。 |
| Europe PMC | `ingest_europepmc()` | 每个 query 的 JSON | 保留既有开放获取筛选；记录开放获取状态和 Europe PMC URL。 |
| ClinicalTrials.gov | `ingest_clinicaltrials()` | 每个 condition 的 JSON | 固定为 `trial_registration`/`other`；记录 NCT、状态、干预、主要终点及可验证的结果状态来源。 |
| 本地资料 | `ingest_local()` | `data/raw/local/` 中的 PDF、Markdown 和种子资料 | 区分教学样本与可验证的本地公开文档，并写入来源定位信息。 |

在线批量采集会为每次运行创建 `data/raw/{source}/{run_id}/`。成功响应在解析前落盘为 `response-*.json` 或 `response-*.xml`，并以 `manifest.json` 记录查询、请求参数、起止时间、原始文件、文档数量和错误。所有原始响应路径均为项目根相对 POSIX 路径。

### 2.3 标准化、去重与质量控制

- `normalize_evidence_level()` 根据文章类型和标题识别 guideline、meta、rct、observational 等等级；
- `merge_docs()` 按 `doc_id` 合并；
- `dedupe_with_stats()` 按 DOI 与归一化标题去重，保留信息更完整的记录，并保存合并血缘；
- ClinicalTrials 登记记录不参与 DOI/标题去重；
- `export_ingest_report()` 输出来源、证据等级、记录类型、年份和缺失字段统计；
- 文档切分后会校验 Chunk ID、文档关联、来源、URL 和 `source_locator`，校验失败时建库会中止。

### 2.4 Wiki 与向量库

- 由主题和标签挑选相关文档，生成 Wiki 导航页；
- Wiki 文档带派生来源信息，但 `citation_eligible=false`；
- Chunk 会继承文档的引用与追溯字段；
- Chroma 将 `citation_eligible`、`record_type`、`source_locator` 作为元数据写入，并将 `provenance` 序列化为 JSON 元数据。

## 3. 当前可用产物

| 产物 | 用途 |
| --- | --- |
| `data/processed/documents.json` | 去重后的基础 EvidenceDoc。 |
| `data/processed/wiki_docs.json` | Wiki 导航文档。 |
| `data/processed/documents_with_wiki.json` | 文档血缘与 Wiki 的完整快照；B 组应从这里读取文档级信息。 |
| `data/processed/ingest_report.md` | 本次采集/规范化质量统计。 |
| `data/processed/store_stats.json` | 当前隔离向量库统计。 |
| `data/chroma-a-validation/` | 本次验证生成的 15-Chunk Chroma 库。 |

## 4. B 组对接指南

### 4.1 使用正确的向量库

当前验证库不是默认 `data/chroma/`。B 组联调前必须在同一终端设置：

```powershell
$env:CHROMA_DIR = 'data/chroma-a-validation'
```

然后再启动 B 组的 smoke demo、API 或 UI；否则程序会打开默认 `data/chroma/`，不能代表本次 A 组验证结果。

### 4.2 引用与检索约束

1. 只对 `citation_eligible=true` 的记录生成对外证据引用。
2. `record_type="trial_registration"` 只能表达登记事实，不得据此表述疗效。
3. Wiki 是导航层；补充证据检索和正式引用应排除 `source="wiki"`。
4. 需要核对来源时，读取 Chunk 或 Chroma metadata 中的 `source_locator` 和 JSON 化的 `provenance`。
5. PubMed、Europe PMC、ClinicalTrials 的在线采集文档均可通过 provenance 中的 run ID 和原始响应路径回溯。

### 4.3 B 组建议验证顺序

```powershell
$env:CHROMA_DIR = 'data/chroma-a-validation'
& .venv\Scripts\python.exe scripts\smoke_demo.py
streamlit run src/app/ui.py
& .venv\Scripts\python.exe scripts\run_eval.py
```

如需重新离线生成 A 组验证库：

```powershell
$env:CHROMA_DIR = 'data/chroma-a-validation'
& .venv\Scripts\python.exe scripts\build_kb.py --skip-live
```

该命令更新 `data/raw/local_docs.json` 和 `data/processed/` 产物，但不访问在线 API，也不会 reset 默认 `data/chroma/`。

## 5. 已验证与未验证边界

已验证：

- 四个在线采集文件通过 AST 语法检查；
- mock HTTP 验证确认三种在线 source 的原始响应逐字节留存、manifest 计数、相对路径、provenance 与旧函数调用兼容；
- 隔离离线建库成功：15 个 Chunk 已写入 Chroma，处理文件均具备当前数据契约字段。

未验证：

- 未使用真实 PubMed、Europe PMC、ClinicalTrials API，以避免真实网络请求；
- 未用本次隔离库执行 B 组 smoke demo、Streamlit 或评测；这些由 B 组联调执行。

## 6. 与《数据字典与 A 组交接》的关系

本文件与 `数据字典与A组交接.md` 有必要但有限的重合。

- 《数据字典与 A 组交接》是稳定的**字段和使用规则**：定义数据契约、记录类型、引用边界及安全建库原则。
- 本文件是当前版本的**实施与交付状态**：说明实现了什么、当前产物在哪里、验证结果是什么、B 组要设置哪个 Chroma 目录以及如何联调。

后续字段语义变更应优先更新《数据字典与 A 组交接》；产物数量、验证 run、建库位置和联调步骤变更应更新本文件。
