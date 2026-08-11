# -*- coding: utf-8 -*-
"""
采集层公共工具：读写与合并 EvidenceDoc。

文末「待完善」签名供队员在本模块内补实现。
"""

from __future__ import annotations

import json
import re
from collections import Counter
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
# 【待完善】采集质量与规范化（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def normalize_evidence_level(raw_type: str, title: str) -> EvidenceLevel:
    """
    【待完善】根据原始文献类型与标题，推断统一证据等级。

    参数:
        raw_type: 数据源返回的原始类型字符串（如 PublicationType、pubType）。
        title: 文献标题，用于关键词兜底判断。

    返回:
        EvidenceLevel: 取值为 rct / meta / guideline / observational / ebook / wiki / other。

    作用:
        让不同数据源的证据等级口径一致，供检索加权与临床展示使用。
    """
    blob = f"{raw_type or ''} {title or ''}".lower()

    # 先判定最高优先级类型，避免 systematic review guideline 被误归为 meta。
    if any(k in blob for k in ["guideline", "practice guideline", "recommendation", "consensus", "statement", "指南", "共识"]):
        return "guideline"
    if any(k in blob for k in ["meta-analysis", "meta analysis", "systematic review", "系统评价", "荟萃", "meta分析"]):
        return "meta"
    if any(k in blob for k in ["randomized", "randomised", "random allocation", "controlled clinical trial", "clinical trial", "rct", "随机", "临床试验"]):
        return "rct"
    if any(k in blob for k in ["cohort", "case-control", "case control", "cross-sectional", "observational", "registry", "real-world", "队列", "病例对照", "横断面", "观察性", "真实世界"]):
        return "observational"
    if any(k in blob for k in ["wiki", "topic page", "主题知识页"]):
        return "wiki"
    if any(k in blob for k in ["book", "ebook", "textbook", "pdf", "markdown", "电子书", "教材"]):
        return "ebook"
    return "other"


def _normalize_doi(doi: str) -> str:
    """把 DOI 清成适合去重的稳定键。"""
    value = (doi or "").strip().lower()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    value = re.sub(r"^doi:\s*", "", value)
    return value.strip(" .。")


def _normalize_title(title: str) -> str:
    """标题归一化：大小写、空白和标点差异不影响去重。"""
    value = re.sub(r"\s+", " ", (title or "").strip().lower())
    value = re.sub(r"[^\w\u4e00-\u9fff]+", "", value)
    return value


def _doc_quality_score(doc: EvidenceDoc) -> int:
    """为重复文档挑选信息更完整的一条。"""
    level_score = {
        "guideline": 700,
        "meta": 650,
        "rct": 600,
        "observational": 450,
        "wiki": 350,
        "ebook": 300,
        "other": 100,
    }
    source_score = {
        "pubmed": 500,
        "europepmc": 450,
        "clinicaltrials": 430,
        "local": 250,
        "wiki": 200,
    }
    return (
        min(len(doc.text or ""), 8000)
        + (1500 if doc.url else 0)
        + (1200 if doc.doi else 0)
        + (300 if doc.year else 0)
        + len(set(doc.tags)) * 80
        + level_score.get(doc.evidence_level, 0)
        + source_score.get(doc.source, 0)
    )


def _merge_duplicate_docs(primary: EvidenceDoc, secondary: EvidenceDoc) -> EvidenceDoc:
    """合并重复记录的补充字段，同时保留主要记录的正文与标题。"""
    tags = sorted(set(primary.tags) | set(secondary.tags))
    extra = dict(secondary.extra or {})
    extra.update(primary.extra or {})
    duplicate_ids = set(extra.get("duplicate_doc_ids") or [])
    duplicate_ids.add(secondary.doc_id)
    if primary.doc_id != secondary.doc_id:
        duplicate_ids.add(primary.doc_id)
    extra["duplicate_doc_ids"] = sorted(duplicate_ids)
    return primary.model_copy(
        update={
            "url": primary.url or secondary.url,
            "doi": primary.doi or secondary.doi,
            "journal": primary.journal or secondary.journal,
            "year": primary.year or secondary.year,
            "tags": tags,
            "extra": extra,
        }
    )


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
    by_key: dict[str, EvidenceDoc] = {}
    order: list[str] = []

    for doc in docs:
        doi = _normalize_doi(doc.doi)
        title_key = _normalize_title(doc.title)
        if doi:
            key = f"doi:{doi}"
        elif len(title_key) >= 12:
            key = f"title:{title_key}"
        else:
            key = f"doc:{doc.doc_id}"

        if key not in by_key:
            by_key[key] = doc
            order.append(key)
            continue

        current = by_key[key]
        if _doc_quality_score(doc) > _doc_quality_score(current):
            by_key[key] = _merge_duplicate_docs(doc, current)
        else:
            by_key[key] = _merge_duplicate_docs(current, doc)

    return [by_key[k] for k in order]


def export_ingest_report(docs: list[EvidenceDoc], out_path: Path) -> Path:
    """
    【待完善】导出采集质量报告（来源分布、年份分布、缺摘要比例等）。

    参数:
        docs: 已采集文档列表。
        out_path: 报告输出路径（建议 md 或 json）。

    返回:
        Path: 实际写入的文件路径。

    作用:
        方便演示与报告中说明「数据从哪来、质量如何」。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(docs)
    by_source = Counter(d.source for d in docs)
    by_level = Counter(d.evidence_level for d in docs)
    by_year = Counter(str(d.year) if d.year else "unknown" for d in docs)
    by_tag = Counter(tag for d in docs for tag in d.tags)
    missing_title = sum(1 for d in docs if not d.title.strip())
    missing_text = sum(1 for d in docs if not d.text.strip())
    missing_url = sum(1 for d in docs if not d.url.strip())
    missing_year = sum(1 for d in docs if d.year is None)
    duplicate_doc_ids = [doc_id for doc_id, n in Counter(d.doc_id for d in docs).items() if n > 1]

    payload = {
        "total": total,
        "by_source": dict(by_source),
        "by_evidence_level": dict(by_level),
        "by_year": dict(sorted(by_year.items())),
        "top_tags": dict(by_tag.most_common(30)),
        "quality": {
            "missing_title": missing_title,
            "missing_text": missing_text,
            "missing_url": missing_url,
            "missing_year": missing_year,
            "duplicate_doc_ids": duplicate_doc_ids,
        },
    }

    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    lines = [
        "# 数据采集质量报告",
        "",
        f"- 文档总数: {total}",
        f"- 缺标题: {missing_title}",
        f"- 缺正文/摘要: {missing_text}",
        f"- 缺 URL: {missing_url}",
        f"- 缺年份: {missing_year}",
        f"- 重复 doc_id 数: {len(duplicate_doc_ids)}",
        "",
        "## 来源分布",
        "",
        "| source | count |",
        "|---|---:|",
    ]
    lines.extend(f"| {k} | {v} |" for k, v in by_source.most_common())
    lines.extend(["", "## 证据等级分布", "", "| evidence_level | count |", "|---|---:|"])
    lines.extend(f"| {k} | {v} |" for k, v in by_level.most_common())
    lines.extend(["", "## 高频标签", "", "| tag | count |", "|---|---:|"])
    lines.extend(f"| {k} | {v} |" for k, v in by_tag.most_common(30))
    lines.extend(["", "## 年份分布", "", "| year | count |", "|---|---:|"])
    lines.extend(f"| {k} | {v} |" for k, v in sorted(by_year.items()))
    if duplicate_doc_ids:
        lines.extend(["", "## 重复 doc_id", ""])
        lines.extend(f"- {x}" for x in duplicate_doc_ids[:100])

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
