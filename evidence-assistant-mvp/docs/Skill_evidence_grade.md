# Skill：evidence-grade（证据分级）

> 固化判断标准（D2AM 材料：Skill=可复用的上岗手册，把"何时拒答、证据分级、怎么组织证据卡"等程序性知识打包）。

## 触发场景

需要判断一篇文献/一条记录的证据等级时（采集、检索加权、临床展示、评估），使用 `src.ingest.normalize_evidence_level`。

## 统一口径

| 等级 | 判定依据（优先级从高到低） |
|---|---|
| guideline | 类型含 guideline / practice guideline / 指南 / consensus |
| meta | meta-analysis / systematic review / umbrella review / 荟萃 / 系统综述 |
| rct | randomized / randomised / clinical trial / controlled trial / 随机对照 / 双盲 |
| observational | cohort / observational / case-control / 队列 / 横断面 / 观察性 |
| ebook | book / ebook / chapter / 手册 |
| wiki | wiki / 维基 |
| other | 以上均未命中 |

## 使用方式

```python
from src.ingest import normalize_evidence_level
level = normalize_evidence_level(raw_type, title)          # 采集时（类型+标题）
from src.ingest import enrich_levels_from_text             # 正文补判（other → 有效等级）
```

## 配套

- 检索加权：`src.kb.weights.EVIDENCE_LEVEL_WEIGHTS`（meta 1.0 > 指南 0.95 > rct 0.9 …）
- 临床赛道期望：指南/系统综述/RCT 级证据，缺失即"标注不足"拒答（规则 2）

## 边界

- 等级判断是**客观规则**，不是模型自由发挥；判不出就标 other，不猜测。
- 演示报告引用"证据等级分布"时，使用 `python scripts/kb_tools.py stats`。
