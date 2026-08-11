# RAG 提示词设计说明（三层架构）

> 依据：SurePrompts《RAG Prompt Engineering: How to Write Prompts That Work With Retrieval-Augmented Generation (2026)》，
> 原文：<https://sureprompts.com/blog/rag-prompt-engineering-guide>。
> 本文档记录本项目（`evidence-assistant-mvp`）按该指南落地的三层提示词设计与实现位置。

---

## 一、设计总览

RAG 提示词分三层，每层解决不同问题：

| 层 | 解决的问题 | 实现位置 | 对应指南要点 |
|---|---|---|---|
| 第 1 层 · 查询改写 | 用户口语与文献语言差距大，单路检索召回不足 | `src/tracks/clinical.py` / `nutrition.py` 的 `rewrite_*_query`；`src/tracks/pipeline.py` 的 `_retrieve_fused` | Query Reformulation、多查询（Multi-Query） |
| 第 2 层 · 系统提示 | 模型不锚定证据、引用格式不稳定、缺失/冲突处理缺失 | `src/generation/answer.py` 的 `generate_answer` 系统提示 | Grounding / Citation / Missing / Conflict / Domain 五要素 |
| 第 3 层 · 综合装配 | 多证据如何组织成结构化、可核对的最终回答 | `pipeline.ask()` 大纲注入 + 参考文献 + 引用校验/修复 + 拒答模板 | Synthesis and Assembly |

---

## 二、第 1 层：查询改写（Query Reformulation）

**目标**：把用户口语问题改写为同时命中英文文献与中文主题页的检索查询。

### 临床赛道（`src/tracks/clinical.py` → `rewrite_clinical_query`）

系统提示：

```text
将用户问题改写为适合检索医学文献的查询。
输出格式（只输出一行，不要解释）：
英文检索式（含疾病/干预/结局/研究类型关键词，可用 AND/OR 连接） || 中文核心关键词（逗号分隔）。
示例：Hypertension AND antihypertensive therapy AND guideline || 高血压, 降压治疗, 指南
```

### 营养赛道（`src/tracks/nutrition.py` → `rewrite_nutrition_query`）

系统提示：

```text
把消费者口语健康问题改写成可检索的证据查询。
输出格式（只输出一行，不要解释）：
英文关键词（饮食模式/营养干预/疾病风险） || 中文关键词（逗号分隔）。
示例：Mediterranean diet AND cardiovascular risk || 地中海饮食, 心血管风险
```

**实际效果示例**（c4 题）：

```text
输入：高血压合并糖尿病时，降压治疗需要关注哪些证据要点？
改写：Hypertension AND diabetes mellitus AND antihypertensive therapy AND evidence || 高血压, 糖尿病, 降压治疗, 证据要点
```

### 配套：多路召回（`src/tracks/pipeline.py` → `_retrieve_fused`）

指南强调查询改写要真正提升召回。本项目实现**双查询融合**：

- 第一路：改写后的检索式（英文文献为主，走 LLM 重排）；
- 第二路：用户原问题（中文关键词，靠 BM25 中文二元切词命中中文 Wiki/种子）；
- 两路结果按 `chunk_id` 融合去重、按分数排序，LLM 重排只对第一路执行以控制成本。

```python
contexts = _retrieve_fused(
    retriever,
    [rewritten, question],   # 改写式 + 原问题两路召回
    top_k=top_k,
    prefer_levels=prefer,
    boost_tags=boost,
)
```

检索侧配套（`src/retrieval/hybrid.py`）：

- **RRF 多路融合**（`reciprocal_rank_fusion`）：向量 / BM25 / 等级限定 BM25 三路排名倒数融合，避免单一召回头部分数淹没相关命中；
- **证据等级感知召回**：临床赛道额外从指南/荟萃/RCT 级语料做一次 BM25 召回；
- **BM25 中文二元切词**：中文整句不再退化为单一词元。

---

## 三、第 2 层：综合生成系统提示（System Prompt for Synthesis）

`src/generation/answer.py` → `generate_answer`，系统消息 = **赛道人格 + 硬性规则 + 证据列表 + 用户问题**。

### 硬性规则（完整原文）

```text
硬性规则：
1. 只能依据给定证据作答，禁止编造文献、PMID、NCT 或链接，不能用训练知识补全。
2. 关键结论句末使用 [n] 引用编号，n 必须来自证据列表。
3. 若证据不足以回答，明确说明证据不足，不要猜测。
4. 若证据只能部分回答，先给出能确认的结论，再明确列出哪些方面缺乏证据支撑，不用推测补全。
5. 证据来源少于 3 个时，用「研究提示」「可能」等弱化表述，不下确定结论。
6. 证据冲突时并列呈现并说明不一致，不选边、不私自修正成唯一答案。
7. 问题超出范围时明确说明边界并拒绝猜测。
8. 文末用「参考文献」列出用到的 [n]。
9. 回答风格：{answer_style}
```

### 五要素对照（指南 → 本项目）

| 指南要素 | 规则编号 | 说明 |
|---|---|---|
| Grounding（锚定证据） | 规则 1 | 只用证据作答、禁止训练知识补全 |
| Citation（引用格式） | 规则 2、8 | 句末 `[n]` 内联引用 + 文末参考文献清单 |
| Missing（缺失处理） | 规则 3、4、5 | 明确证据不足、部分回答先确认再标注缺口、弱化表述 |
| Conflict（冲突处理） | 规则 6 | 并列呈现并说明不一致，不选边 |
| Domain（领域规则） | 人格 + 风格 + 赛道四段式 | 临床：结论→证据等级→关键研究→局限；营养：通俗结论→证据一句话→你可以怎么做→何时就医 |

---

## 四、第 3 层：综合与装配（Synthesis and Assembly）

### 1. 结构化大纲注入（`pipeline.ask()`）

生成前按赛道组装 `effective_style`，把四段式结构写进提示，保证输出可预期：

```text
# 临床追加
回答必须按「结论 → 证据等级 → 关键研究/指南 → 局限」四段组织，结论句前先汇总所依据的证据等级。

# 营养追加
回答必须按「通俗结论 → 证据一句话 → 你可以怎么做 → 何时就医」四段组织，用词口语化、避免堆砌术语。
```

大纲同时以结构化数据挂到响应（`citation_check.clinical_outline` / `nutrition_outline`），供界面分区渲染与演示讲解。

### 2. 参考文献自动补全（`format_reference_section`）

模型漏写「参考文献」时，按 `[n] 标题 · 来源 · 年份 · 链接` 格式自动补齐，保证引用可核对。

### 3. 引用校验 + LLM 修复

- `verify_citations`：检测无效编号、编造 PMID/NCT/文档 ID；
- `repair_answer_with_valid_cites`：在线模式用 DeepSeek 结构化重写（JSON 输出），校验通过才采用；离线/失败回退规则式清理；
- 校验不过时追加「无法核实」提示（`strip_invalid_claims`），不静默放行。

### 4. 合格拒答三要素（`_build_qualified_refusal`）

拒答不等于"我不能回答"，必须说明：

```text
我能确认的是：{已检索到什么（条数 + 来源分布）}；但{缺什么（越界 / 证据不足 / 缺高质量证据）}。
如需继续：{建议补查什么}。
```

### 5. 产品取舍说明

指南建议"文末不另设参考文献，内联引用即可"。本项目**有意保留文末参考文献**：

- 医疗证据产品需要可核对的完整清单（评委/用户可逐条回查）；
- 界面证据面板与引用编号体系依赖该清单；
- 内联 `[n]` + 文末清单并存，兼顾可读性与可核对性。

---

## 五、验证结果（真实 DeepSeek + 230 条知识库）

| 检查项 | 结果 |
|---|---|
| 临床题集（`scripts/check_clinical.py`） | 5/5：引用校验通过、大纲齐全、要点覆盖 1.0 |
| 营养题集（`scripts/check_nutrition.py`） | 5/5：同上 |
| 越界/药量拒答 | 带原因 + 改进问法建议 |
| 在线补检索（`use_live_tools=True`） | 7 条证据，来源含 pubmed/clinicaltrials，引用校验通过 |
| 冒烟测试 | `smoke_demo OK` |

> 注：检索向量当前为哈希 embedding + BM25 兜底（DeepSeek 无 embedding 接口、本地模型下载受网络限制）。
> 提示层设计不受影响；接入真实 embedding 后召回质量会进一步提升。

---

## 六、代码位置索引

| 模块 | 函数 | 对应层 |
|---|---|---|
| `src/tracks/clinical.py` | `rewrite_clinical_query` | 第 1 层 |
| `src/tracks/nutrition.py` | `rewrite_nutrition_query` | 第 1 层 |
| `src/tracks/pipeline.py` | `_retrieve_fused` / `ask` | 第 1 层（多路召回）+ 第 3 层（装配） |
| `src/retrieval/hybrid.py` | `reciprocal_rank_fusion` / `retrieve` / `_bm25_search` | 第 1 层支撑 |
| `src/generation/answer.py` | `generate_answer` / `format_reference_section` | 第 2 层 + 第 3 层 |
| `src/tools/cite_check.py` | `verify_citations` / `repair_answer_with_valid_cites` / `strip_invalid_claims` | 第 3 层 |
| `src/tracks/pipeline.py` | `_build_qualified_refusal` / `_refusal_suggestions` | 第 3 层 |
