# -*- coding: utf-8 -*-
"""
采集层公共工具：读写与合并 EvidenceDoc。

文末「待完善」签名供队员在本模块内补实现。
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

from src.models import EvidenceDoc, EvidenceLevel


def save_docs(docs: Iterable[EvidenceDoc], path: Path) -> int:
    """
    将证据文档列表保存为 JSON 文件。

    参数:
        docs: 可迭代的 EvidenceDoc。
        path: 输出文件路径。

    返回:
        int: 实际写入条数。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [d.model_dump() for d in docs]
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(items)


def load_docs(path: Path) -> list[EvidenceDoc]:
    """
    从 JSON 文件加载证据文档。

    参数:
        path: JSON 路径。

    返回:
        list[EvidenceDoc]: 文档列表；文件不存在时返回空列表。
    """
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvidenceDoc.model_validate(x) for x in raw]


def merge_docs(*doc_lists: list[EvidenceDoc]) -> list[EvidenceDoc]:
    """
    按 doc_id 去重合并多组文档（先出现者优先保留）。

    参数:
        *doc_lists: 多组 EvidenceDoc 列表。

    返回:
        list[EvidenceDoc]: 合并结果。
    """
    seen: set[str] = set()
    out: list[EvidenceDoc] = []
    for docs in doc_lists:
        for d in docs:
            if d.doc_id in seen:
                continue
            seen.add(d.doc_id)
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# 采集质量与规范化
# （normalize_evidence_level 已实现；其余任务仍为待完善签名，见各自注释）
# ---------------------------------------------------------------------------


def normalize_evidence_level(raw_type: str, title: str) -> EvidenceLevel:
    """
    根据原始文献类型与标题，推断统一证据等级。

    参数:
        raw_type: 数据源返回的原始类型字符串（如 PublicationType、pubType）。
        title: 文献标题，用于关键词兜底判断。

    返回:
        EvidenceLevel: 取值为 rct / meta / guideline / observational / ebook / wiki / other。

    作用:
        让不同数据源的证据等级口径一致，供检索加权与临床展示使用。
    """
    blob = f"{raw_type} {title}".lower()
    if any(k in blob for k in ("guideline", "practice guideline", "指南", "consensus")):
        return "guideline"
    if any(k in blob for k in ("meta-analysis", "meta analysis", "systematic review", "荟萃")):
        return "meta"
    if any(k in blob for k in ("randomized", "randomised", "clinical trial", "controlled trial", "随机")):
        return "rct"
    if any(k in blob for k in ("cohort", "observational", "case-control", "cross-sectional", "case series", "case report")):
        return "observational"
    if any(k in blob for k in ("book", "ebook", "chapter", "手册")):
        return "ebook"
    if any(k in blob for k in ("wiki", "维基")):
        return "wiki"
    return "other"


def dedupe_by_doi_or_title(docs: list[EvidenceDoc]) -> list[EvidenceDoc]:
    """
    【待完善】按 DOI 优先、其次标题归一化，对证据文档强去重。

    参数:
        docs: 合并后的证据文档列表。

    返回:
        list[EvidenceDoc]: 去重后的文档列表（保留信息更完整的一条）。

    作用:
        减少同一文献多源重复进入知识库，降低检索噪声。
    """
    raise NotImplementedError("待队员实现：dedupe_by_doi_or_title")


def export_ingest_report(docs: list[EvidenceDoc], out_path: Path) -> Path:
    """
    导出采集质量报告（来源分布、年份分布、缺摘要比例等）。

    参数:
        docs: 已采集文档列表。
        out_path: 报告输出路径（建议 md 或 json）。

    返回:
        Path: 实际写入的文件路径。

    作用:
        方便演示与报告中说明「数据从哪来、质量如何」。
    """
    total = len(docs)
    sources = Counter(d.source for d in docs)
    years = Counter(d.year for d in docs)
    levels = Counter(d.evidence_level for d in docs)
    tags = Counter(t for d in docs for t in d.tags)
    no_text = [d.doc_id for d in docs if not (d.text or "").strip()]
    no_year = sum(1 for d in docs if d.year is None)
    with_doi = sum(1 for d in docs if d.doi)

    def pct(n: int) -> str:
        return f"{n / total * 100:.1f}%" if total else "0.0%"

    known_years = [y for y in years if y is not None]
    year_range = (
        f"{min(known_years)} ~ {max(known_years)}" if known_years else "未知"
    )

    lines = [
        "# 采集质量报告",
        "",
        f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M}",
        f"- 文档总数: {total}",
        f"- 数据源数: {len(sources)}",
        f"- 年份覆盖: {year_range}",
        f"- 缺摘要比例: {len(no_text)}/{total} ({pct(len(no_text))})",
        f"- 缺年份比例: {no_year}/{total} ({pct(no_year)})",
        f"- DOI 覆盖率: {with_doi}/{total} ({pct(with_doi)})",
        "",
        "## 来源分布",
        "",
        "| 来源 | 条数 | 占比 |",
        "|---|---|---|",
    ]
    lines += [f"| {s} | {n} | {pct(n)} |" for s, n in sources.most_common()]
    lines += [
        "",
        "## 年份分布",
        "",
        "| 年份 | 条数 | 占比 |",
        "|---|---|---|",
    ]
    lines += [
        f"| {y if y is not None else '未知'} | {n} | {pct(n)} |"
        for y, n in sorted(years.items(), key=lambda kv: (kv[0] is None, kv[0]))
    ]
    lines += [
        "",
        "## 证据等级分布",
        "",
        "| 等级 | 条数 | 占比 |",
        "|---|---|---|",
    ]
    lines += [
        f"| {lv} | {n} | {pct(n)} |" for lv, n in levels.most_common()
    ]
    lines += [
        "",
        "## 标签分布（Top 10）",
        "",
        "| 标签 | 条数 | 占比 |",
        "|---|---|---|",
    ]
    lines += [
        f"| {t} | {n} | {pct(n)} |" for t, n in tags.most_common(10)
    ]
    if no_text:
        lines += ["", "## 缺摘要文档", ""]
        lines += [f"- {doc_id}" for doc_id in no_text[:20]]
        if len(no_text) > 20:
            lines.append(f"- …（共 {len(no_text)} 条）")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
