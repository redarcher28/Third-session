# -*- coding: utf-8 -*-
"""
采集层公共工具：读写与合并 EvidenceDoc。

文末「待完善」签名供队员在本模块内补实现。
"""

from __future__ import annotations

import json
import re
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


def _norm_doi(doi: str) -> str:
    """DOI 归一化：去前缀、小写、去空白，用于跨源比对。"""
    d = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.strip()


def _norm_title(title: str) -> str:
    """标题归一化：小写、去冠词、去标点、压缩空白，用于标题级比对。"""
    t = re.sub(r"^(the|a|an)\s+", "", title.lower().strip())
    t = re.sub(r"\s+", " ", t)
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", t)


def _completeness(d: EvidenceDoc) -> int:
    """信息完整度：正文越长、元数据越全，得分越高，去重时保留高分者。"""
    score = len((d.text or "").strip())
    if d.year:
        score += 1000
    if d.url:
        score += 500
    if d.journal:
        score += 500
    if d.doi:
        score += 300
    score += len(d.tags) * 100
    return score


def _merge_fields(keep: EvidenceDoc, other: EvidenceDoc) -> EvidenceDoc:
    """
    把 other 的非空字段融合进 keep（取并集，优先 keep 的既有值）。

    优化点：去重不再「二选一丢信息」，而是保留完整度高的同时，
    把被丢弃文档的 url/journal/year/doi/正文/tags/extra 补进胜出文档。
    """
    if not keep.url and other.url:
        keep.url = other.url
    if not keep.journal and other.journal:
        keep.journal = other.journal
    if not keep.doi and other.doi:
        keep.doi = other.doi
    if keep.year is None and other.year is not None:
        keep.year = other.year
    if not (keep.text or "").strip() and (other.text or "").strip():
        keep.text = other.text
    for t in other.tags:
        if t not in keep.tags:
            keep.tags.append(t)
    for k, v in (other.extra or {}).items():
        if v and k not in (keep.extra or {}):
            keep.extra[k] = v
    return keep


def dedupe_by_doi_or_title(docs: list[EvidenceDoc]) -> list[EvidenceDoc]:
    """
    按 DOI 优先、其次标题归一化，对证据文档强去重。

    优化点：
        - 标题归一化额外剥离 the/a/an 冠词，跨源标题更易对齐；
        - 去重胜出文档会融合被丢弃文档的非空字段（url/journal/year/DOI/tags），
          「保留信息更完整的一条」从二选一升级为取并集。

    参数:
        docs: 合并后的证据文档列表。

    返回:
        list[EvidenceDoc]: 去重后的文档列表（保留信息更完整的一条）。

    作用:
        减少同一文献多源重复进入知识库，降低检索噪声。
    """
    seen: dict[str, EvidenceDoc] = {}
    for d in docs:
        if d.doi:
            key = f"doi:{_norm_doi(d.doi)}"
        else:
            key = f"title:{_norm_title(d.title)}"
        if key in ("doi:", "title:"):  # 无有效 DOI/标题，不参与去重
            key = f"id:{d.doc_id}"
        prev = seen.get(key)
        if prev is None:
            seen[key] = d
        elif _completeness(d) > _completeness(prev):
            seen[key] = _merge_fields(d, prev)  # 新文档胜出，融合旧文档字段
        else:
            seen[key] = _merge_fields(prev, d)  # 旧文档胜出，融合新文档字段
    return list(seen.values())


# 正文关键词 → 证据等级（高证据优先：meta > guideline > rct > observational）
_TEXT_LEVEL_KEYS: dict[str, list[str]] = {
    "meta": ["meta-analysis", "meta analysis", "systematic review", "umbrella review", "荟萃", "系统综述", "伞形综述"],
    "guideline": ["guideline", "指南", "recommendation", "consensus"],
    "rct": ["randomized", "randomised", "randomized controlled", "随机对照", "double-blind", "双盲"],
    "observational": ["cohort", "observational", "case-control", "队列", "横断面", "观察性"],
}


def enrich_levels_from_text(docs: list[EvidenceDoc]) -> list[EvidenceDoc]:
    """
    对 evidence_level=other 的文档，用标题+正文关键词补判证据等级（原地更新）。

    创新点：标题启发式对真实文献命中率低（500 合集中大量 Journal Article 标题
    不含类型词），正文摘要几乎必然出现 randomized/cohort/meta-analysis 等词，
    用正文补判可把 other 占比显著压低，让检索加权真正生效。
    """
    for d in docs:
        if d.evidence_level != "other":
            continue
        blob = f"{d.title} {d.text}"[:3000].lower()
        for level, keys in _TEXT_LEVEL_KEYS.items():
            if any(k in blob for k in keys):
                d.evidence_level = level  # type: ignore[assignment]
                break
    return docs


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
