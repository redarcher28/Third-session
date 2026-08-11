# -*- coding: utf-8 -*-
"""
在线补检索工具：问答时可临时从 PubMed / 临床试验拉取少量证据。
"""

from __future__ import annotations

from src.ingest.clinicaltrials import search_trials
from src.ingest.pubmed import fetch_pubmed_docs, search_pmids
from src.models import EvidenceDoc


def search_pubmed(query: str, retmax: int = 5) -> list[EvidenceDoc]:
    """
    在线检索 PubMed（供 Agent/流水线补证据）。

    参数:
        query: 检索词。
        retmax: 最多条数。

    返回:
        list[EvidenceDoc]: 文献列表。
    """
    pmids = search_pmids(query, retmax=retmax)
    return fetch_pubmed_docs(pmids)


def search_clinical_trials(condition: str, page_size: int = 5) -> list[EvidenceDoc]:
    """
    在线检索 ClinicalTrials.gov。

    参数:
        condition: 病种/条件。
        page_size: 条数。

    返回:
        list[EvidenceDoc]: 试验列表。
    """
    return search_trials(condition, page_size=page_size)


# ---------------------------------------------------------------------------
# 在线补检索编排
# ---------------------------------------------------------------------------


def merge_live_and_offline_docs(
    offline: list[EvidenceDoc],
    live: list[EvidenceDoc],
    max_total: int = 8,
) -> list[EvidenceDoc]:
    """
    合并离线库证据与在线补检索结果并去重截断。

    参数:
        offline: 知识库侧文档/块对应的文档。
        live: 在线 PubMed/Trials 结果。
        max_total: 合并后上限。

    返回:
        list[EvidenceDoc]: 合并去重后的列表。

    作用:
        保证「默认离线稳定 + 可选在线增强」的可控融合。
    """
    if max_total <= 0:
        return []
    merged: list[EvidenceDoc] = []
    seen: set[str] = set()
    for doc in [*offline, *live]:
        if doc.doc_id in seen:
            continue
        seen.add(doc.doc_id)
        merged.append(doc)
        if len(merged) >= max_total:
            break
    return merged


def should_trigger_live_search(question: str, offline_hit_count: int) -> bool:
    """
    判断是否有必要触发在线补检索。

    参数:
        question: 用户问题。
        offline_hit_count: 离线检索命中条数。

    返回:
        bool: True 表示建议启用在线工具。

    作用:
        避免每次都打外网 API，节省配额并保持演示稳定。
    """
    if offline_hit_count < 3:
        return True
    text = question.lower()
    trigger_hints = [
        "最新", "近五年", "2023", "2024", "2025", "2026",
        "recent", "latest", "update", "guideline",
    ]
    return any(hint in text for hint in trigger_hints)
