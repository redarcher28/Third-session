# -*- coding: utf-8 -*-
"""
Europe PMC 采集模块。

职责：补充开放获取文献摘要，映射为统一 EvidenceDoc。
"""

from __future__ import annotations

import logging

import httpx

from src.ingest import IngestRun, normalize_evidence_level, start_ingest_run
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


def search_europepmc(query: str, page_size: int = 15, *, run: IngestRun | None = None) -> list[EvidenceDoc]:
    """Search Europe PMC and optionally retain the successful JSON response."""
    params = {"query": query, "format": "json", "pageSize": page_size, "resultType": "core"}
    with httpx.Client(timeout=60) as client:
        response = client.get(API, params=params)
        response.raise_for_status()
        raw_path = run.save_response(response.content, "json") if run else ""
        payload = response.json()

    results = payload.get("resultList", {}).get("result", []) or []
    docs: list[EvidenceDoc] = []
    for item in results:
        record_source = item.get("source", "MED")
        ext_id = item.get("id") or item.get("pmid") or item.get("doi") or ""
        title = item.get("title") or ""
        abstract = item.get("abstractText") or title
        year = item.get("pubYear")
        year_i = int(year) if year and str(year).isdigit() else None
        pmid = item.get("pmid")
        europepmc_url = f"https://europepmc.org/article/{record_source}/{ext_id}"
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else europepmc_url
        is_open_access = str(item.get("isOpenAccess") or "").upper() == "Y"
        provenance = {}
        if run is not None:
            provenance = {
                "run_id": run.run_id,
                "source": "europepmc",
                "query": query,
                "open_access": is_open_access,
                "europepmc_url": europepmc_url,
                "retrieved_at": run.started_at,
                "raw_response_path": raw_path,
            }
        docs.append(EvidenceDoc(
            doc_id=f"epmc:{record_source}:{ext_id}", source="europepmc", title=title,
            text=abstract[:8000], year=year_i, url=url, tags=_tags(f"{title} {abstract}"),
            evidence_level=normalize_evidence_level(item.get("pubType") or "", title),
            journal=item.get("journalTitle") or "", doi=item.get("doi") or "",
            citation_eligible=True, record_type="published_evidence",
            source_locator=f"Europe PMC:{record_source}:{ext_id}", provenance=provenance,
            extra={"isOpenAccess": item.get("isOpenAccess") or "", "inPMC": item.get("inPMC") or "", "pubType": item.get("pubType") or ""},
        ))
    return docs

DEFAULT_QUERIES = [
    "hypertension guidelines open access",
    "Mediterranean diet cardiovascular systematic review",
    "dietary sodium hypertension",
    "statin therapy hyperlipidemia guidelines",
]


def ingest_europepmc(queries: list[str] | None = None, page_size: int = 12) -> list[EvidenceDoc]:
    """Batch-ingest Europe PMC while retaining raw query responses."""
    queries = queries or DEFAULT_QUERIES
    try:
        run = start_ingest_run("europepmc", queries, {"page_size": page_size})
    except Exception as exc:
        logger.warning("EuropePMC run initialization failed: %s", exc)
        return []

    docs: list[EvidenceDoc] = []
    try:
        for query in queries:
            try:
                query_docs = filter_open_access_only(search_europepmc(query, page_size=page_size, run=run))
            except Exception as exc:
                message = f"source=europepmc query={query!r} {type(exc).__name__}: {exc}"
                run.errors.append(message)
                logger.warning("EuropePMC query failed: %s", message)
                continue
            logger.info("EuropePMC query=%r -> %d (after OA filter)", query, len(query_docs))
            docs.extend(query_docs)
    except Exception as exc:
        message = f"source=europepmc {type(exc).__name__}: {exc}"
        run.errors.append(message)
        logger.warning("EuropePMC fetch failed: %s", message)
    finally:
        run.finalize(docs)
    return docs

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
