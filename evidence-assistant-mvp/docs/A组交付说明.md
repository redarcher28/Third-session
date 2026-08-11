# A 组交付说明（B 组对接文档）

> 数据与知识库组产出：从采集到向量入库的完整链路，B 组只需消费下述接口与文件。

## 1. 数据流总览

```
采集层 src/ingest/                知识库层 src/kb/
─────────────────────────        ─────────────────────────
本地种子/PDF/Markdown   ─┐
PubMed API (PICO 检索式) ─┤→ 统一 EvidenceDoc
ClinicalTrials API      ─┤→ dedupe_by_doi_or_title(融合)
EuropePMC API (OA 过滤)  ─┘→ documents.json
                                   ↓
                    generate_wiki_pages → wiki/*.md → documents_with_wiki.json
                                   ↓
              docs_to_chunks → merge_tiny_chunks → validate_chunk_traceability
                                   ↓
                          Chroma(evidence_chunks) ← rebuild(增量/剪枝)
```

## 2. 交付文件与函数速查

| 模块 | 函数 | 用途 |
|---|---|---|
| `src.ingest.normalize_evidence_level` | 统一 7 级证据等级 | 检索加权的**等级字段来源** |
| `src.ingest.dedupe_by_doi_or_title` | DOI/标题去重+字段融合 | 保证同一文献只留一条完整记录 |
| `src.ingest.pubmed.build_mesh_aware_query` | PICO 结构化检索式 | 自定义采集时构造查询 |
| `src.ingest.clinicaltrials.extract_trial_primary_outcome` | 主要结局提取 | 回答「结局」类问题 |
| `src.kb.store.EvidenceStore.query` | 语义检索（支持 `source=` 过滤） | **B 组检索主入口** |
| `src.kb.store.export_store_stats` | 知识库统计（JSON+MD） | 报告/演示数据 |
| `src.kb.store.rebuild_collection_from_processed` | 全量/增量重建 | 改切分参数后秒级重建 |
| `src.kb.wiki.select_wiki_then_chunks` | **Wiki 优先两级检索** | 建议接进 `pipeline.ask()` |
| `src.kb.wiki.refresh_single_wiki_page` | 单页刷新 | 小步维护主题页 |
| `src.kb.weights.*` | 证据等级权重+时效衰减 | **B 组 `score_evidence_priority` 直接调用** |

## 3. B 组待完善任务的现成铺垫

| B 组任务 | A 组已备好 |
|---|---|
| `score_evidence_priority` | `weights.combined_priority(level, year)` —— 等级×时效组合分，`weights.EVIDENCE_LEVEL_WEIGHTS` 表 |
| `filter_by_year_range` | chunk 元数据已带 `year`（未知为 -1），直接过滤即可 |
| `explain_retrieval` | `store.query` 返回 `distance` + 全量元数据（doc_id/url/level） |
| `diversify_by_source` | chunk 元数据已带 `source`（pubmed/clinicaltrials/europepmc/local/wiki） |
| `reciprocal_rank_fusion` | `store.all_chunks_for_bm25()` 导出 BM25 语料 + `store.query()` 向量结果 |
| 引用可回查 | `validate_chunk_traceability` 已在 build 门槛断言：doc_id/url/标题齐全 |

## 4. 字段约定（改前必须先同步）

```python
evidence_level:  "rct" | "meta" | "guideline" | "observational" | "ebook" | "wiki" | "other"
source:          "pubmed" | "clinicaltrials" | "europepmc" | "local" | "wiki"
extra:           clinicaltrials 含 primary_outcome；europepmc 含 isOpenAccess/inPMC
```

⚠️ `EvidenceDoc`/`Chunk` 为两组共享模型，改字段先对齐（见 `docs/任务分工.md`）。

## 5. 命令速查（B 组与演示直接复制）

```bash
# 构建（含报告+统计）
python scripts/build_kb.py --skip-live --stats

# 只重新建库（不联网、不重新生成 wiki）
python scripts/kb_tools.py rebuild
python scripts/kb_tools.py rebuild --incremental   # 指纹跳过+剪枝

# 知识库统计
python scripts/kb_tools.py stats

# 刷新单个主题页（先列出可选主题）
python scripts/kb_tools.py refresh-wiki
python scripts/kb_tools.py refresh-wiki sodium-hypertension
```

## 6. Wiki 优先检索接入示例（供 pipeline 参考）

```python
from src.kb.wiki import select_wiki_then_chunks

contexts = select_wiki_then_chunks(rewritten_query, wiki_k=2, chunk_k=5)
# 返回项带 kind: "wiki"（总览）| "evidence"（原文支撑），UI 可分组渲染
```
