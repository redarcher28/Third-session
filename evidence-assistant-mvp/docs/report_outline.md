# 项目报告提纲

## 1. 背景与赛道

- 对标 OpenEvidence：问题 → 证据 → 带引用回答
- 本项目同时覆盖三条赛道：临床助手、营养助手、RAG vs 通用大模型评测

## 2. 数据来源与方法

- PubMed / ClinicalTrials.gov / Europe PMC / 本地种子与 PDF
- 统一 EvidenceDoc schema → 切分 → Chroma 向量库 + BM25 混合检索
- LLM Wiki 主题页作为可维护知识旁路
- 生成约束：仅基于检索证据、`[n]` 引用、拒答、引用校验

## 3. 系统演示

- 赛道一：临床结构化回答 + 证据面板
- 赛道二：通俗科普改写 + 同一知识库溯源
- 赛道三：`data/eval/results/benchmark_summary.md` 指标与典型 case

## 4. 初步评估结果

- 填写：假引用率、引用覆盖率、要点覆盖、拒答率
- 讨论：RAG 何时更好、检索差时是否拖累、假引用是否下降

## 5. 局限、伦理与后续

- 非诊疗、无真实患者数据、引用需人工复核
- 窄领域语料、离线占位模式限制
- 后续：更大语料、更强重排、Ragas、MCP 工具化

## 启动命令备忘

```bash
python scripts/build_kb.py --skip-live
streamlit run src/app/ui.py
python scripts/run_eval.py
```
