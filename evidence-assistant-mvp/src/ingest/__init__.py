# -*- coding: utf-8 -*-
"""
采集层公共工具：读写与合并 EvidenceDoc。

文末「待完善」签名供队员在本模块内补实现。
"""

from __future__ import annotations

import json
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
    raise NotImplementedError("待队员实现：normalize_evidence_level")


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
    【待完善】导出采集质量报告（来源分布、年份分布、缺摘要比例等）。

    参数:
        docs: 已采集文档列表。
        out_path: 报告输出路径（建议 md 或 json）。

    返回:
        Path: 实际写入的文件路径。

    作用:
        方便演示与报告中说明「数据从哪来、质量如何」。
    """
    raise NotImplementedError("待队员实现：export_ingest_report")
