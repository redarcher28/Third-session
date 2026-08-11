# -*- coding: utf-8 -*-
"""
Europe PMC 采集模块。

职责：补充开放获取文献摘要，映射为统一 EvidenceDoc。
"""

from __future__ import annotations

import logging

import httpx

from src.ingest import normalize_evidence_level
from src.models import EvidenceDoc

logger = logging.getLogger(__name__)

API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _tags(text: str) -> list[str]:
    mapping = {
        "hypertension": ["hypertension", "blood pressure", "高血压"],
        "hyperlipidemia": ["lipid", "cholesterol", "血脂"],
        "diabetes": ["diabetes", "糖尿病"],
        "cardiovascular": ["cardiovascular", "心血管"],
        "diet": ["diet", "nutrition", "sodium", "饮食"],
        "mediterranean": ["mediterranean", "地中海"],
    }
    lower = text.lower()
    return [t for t, keys in mapping.items() if any(k.lower() in lower for k in keys)]


def search_europepmc(query: str, page_size: int = 15) -> list[EvidenceDoc]:
    """
    单查询检索 Europe PMC。

    参数:
        query: 检索词。
        page_size: 返回条数。

    返回:
        list[EvidenceDoc]: 文献摘要列表。
    """
    params = {
        "query": query,
        "format": "json",
        "pageSize": page_size,
        "resultType": "core",
    }
    with httpx.Client(timeout=60) as client:
        r = client.get(API, params=params)
        r.raise_for_status()
        payload = r.json()
    results = payload.get("resultList", {}).get("result", []) or []
    docs: list[EvidenceDoc] = []
    for item in results:
        source = item.get("source", "MED")
        ext_id = item.get("id") or item.get("pmid") or item.get("doi") or ""
        title = item.get("title") or ""
        abstract = item.get("abstractText") or title
        year = item.get("pubYear")
        year_i = int(year) if year and str(year).isdigit() else None
        pmid = item.get("pmid")
        url = (
            f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            if pmid
            else f"https://europepmc.org/article/{source}/{ext_id}"
        )
        doc_id = f"epmc:{source}:{ext_id}"
        level = normalize_evidence_level(item.get("pubType") or "", title)
        blob = f"{title} {abstract}"
        docs.append(
            EvidenceDoc(
                doc_id=doc_id,
                source="europepmc",
                title=title,
                text=abstract[:8000],
                year=year_i,
                url=url,
                tags=_tags(blob),
                evidence_level=level,  # type: ignore[arg-type]
                journal=item.get("journalTitle") or "",
                doi=item.get("doi") or "",
                extra={
                    "isOpenAccess": item.get("isOpenAccess") or "",
                    "inPMC": item.get("inPMC") or "",
                    "pubType": item.get("pubType") or "",
                },
            )
        )
    return docs


DEFAULT_QUERIES = [
    "hypertension guidelines open access",
    "Mediterranean diet cardiovascular systematic review",
    "dietary sodium hypertension",
    "statin therapy hyperlipidemia guidelines",
]


def ingest_europepmc(
    queries: list[str] | None = None,
    page_size: int = 12,
) -> list[EvidenceDoc]:
    """
    多查询批量采集 Europe PMC。

    参数:
        queries: 查询列表；None 用默认查询。
        page_size: 每查询条数。

    返回:
        list[EvidenceDoc]: 采集结果。
    """
    queries = queries or DEFAULT_QUERIES
    out: list[EvidenceDoc] = []
    try:
        for q in queries:
            docs = filter_open_access_only(search_europepmc(q, page_size=page_size))
            logger.info("EuropePMC query=%r -> %d (after OA filter)", q, len(docs))
            out.extend(docs)
    except Exception as e:
        logger.warning("EuropePMC fetch failed: %s", e)
        return []
    return out


# ---------------------------------------------------------------------------
# Europe PMC 开放获取过滤
# ---------------------------------------------------------------------------


def filter_open_access_only(docs: list[EvidenceDoc]) -> list[EvidenceDoc]:
    """
    仅保留开放获取 / 可公开引用的 Europe PMC 条目。

    创新点：
        - 白名单优先：isOpenAccess=Y 直接保留；
        - 来源回溯兜底：非 OA 但只要被 PMC 收录且带 DOI/URL 的也保留，
          保证演示引用能点开全文或至少公开摘要；
        - 附带丢弃计数日志，方便评估语料质量损耗。

    参数:
        docs: Europe PMC 采集结果。

    返回:
        list[EvidenceDoc]: 过滤后的列表。

    作用:
        保证演示引用尽可能可点开全文或摘要，降低版权风险叙述负担。
    """
    kept: list[EvidenceDoc] = []
    dropped = 0
    for d in docs:
        extra = d.extra or {}
        is_oa = str(extra.get("isOpenAccess", "N")).upper() == "Y"
        in_pmc = str(extra.get("inPMC", "")).upper() == "Y"
        if is_oa or (in_pmc and (d.doi or d.url)):
            kept.append(d)
        else:
            dropped += 1
    if dropped:
        logger.info("EuropePMC OA filter dropped %d/%d docs", dropped, len(docs))
    return kept
