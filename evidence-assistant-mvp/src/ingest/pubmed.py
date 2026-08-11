# -*- coding: utf-8 -*-
"""
PubMed 采集模块（NCBI E-utilities）。

职责：按查询搜索 PMID，再批量拉取摘要，映射为统一 EvidenceDoc。
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from src.config import get_settings
from src.ingest import IngestRun, normalize_evidence_level, start_ingest_run
from src.models import EvidenceDoc

logger = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _params(**extra: Any) -> dict[str, Any]:
    """组装 E-utilities 公共查询参数（含 email / api_key）。"""
    s = get_settings()
    p: dict[str, Any] = {"tool": "evidence_assistant_mvp", "email": s.ncbi_email}
    if s.ncbi_api_key:
        p["api_key"] = s.ncbi_api_key
    p.update(extra)
    return p


def search_pmids(query: str, retmax: int = 20, *, run: IngestRun | None = None) -> list[str]:
    """Search PubMed and optionally retain the successful ESearch response."""
    with httpx.Client(timeout=60) as client:
        response = client.get(f"{EUTILS}/esearch.fcgi", params=_params(db="pubmed", term=query, retmax=retmax, retmode="json"))
        response.raise_for_status()
        if run is not None:
            run.save_response(response.content, "json")
        ids = response.json().get("esearchresult", {}).get("idlist", [])
    return list(ids)

def _text(el: ET.Element | None, path: str = "") -> str:
    if el is None:
        return ""
    node = el.find(path) if path else el
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def fetch_pubmed_docs(pmids: list[str], *, run: IngestRun | None = None) -> list[EvidenceDoc]:
    """Fetch PubMed records in batches, retaining each successful EFetch XML."""
    if not pmids:
        return []
    docs: list[EvidenceDoc] = []
    pmid_queries: dict[str, list[str]] = getattr(run, "pmid_queries", {})
    with httpx.Client(timeout=90) as client:
        for i in range(0, len(pmids), 50):
            batch = pmids[i : i + 50]
            try:
                response = client.get(f"{EUTILS}/efetch.fcgi", params=_params(db="pubmed", id=",".join(batch), retmode="xml"))
                response.raise_for_status()
                raw_path = run.save_response(response.content, "xml") if run else ""
                root = ET.fromstring(response.content)
            except Exception as exc:
                message = f"source=pubmed batch={batch!r} {type(exc).__name__}: {exc}"
                if run is not None:
                    run.errors.append(message)
                logger.warning("PubMed EFetch batch failed: %s", message)
                continue
            for article in root.findall(".//PubmedArticle"):
                pmid = _text(article, ".//PMID")
                title = _text(article, ".//ArticleTitle")
                abstract_parts = ["".join(node.itertext()).strip() for node in article.findall(".//Abstract/AbstractText")]
                abstract = "\n".join(part for part in abstract_parts if part)
                year_text = _text(article, ".//PubDate/Year") or _text(article, ".//PubDate/MedlineDate")[:4]
                year = int(year_text) if year_text.isdigit() else None
                journal = _text(article, ".//Journal/Title")
                doi = ""
                for article_id in article.findall(".//ArticleId"):
                    if article_id.get("IdType") == "doi":
                        doi = (article_id.text or "").strip()
                pub_types = ["".join(pub_type.itertext()).strip() for pub_type in article.findall(".//PublicationType")]
                if not abstract:
                    abstract = title
                provenance = {}
                if run is not None:
                    provenance = {"run_id": run.run_id, "source": "pubmed", "pmid": pmid, "doi": doi,
                                  "article_types": pub_types, "queries": list(pmid_queries.get(pmid, [])),
                                  "retrieved_at": run.started_at, "raw_response_paths": [raw_path]}
                docs.append(EvidenceDoc(doc_id=f"pmid:{pmid}", source="pubmed", title=title or f"PMID {pmid}",
                    text=abstract, year=year, url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", tags=_tags_from_text(f"{title} {abstract}"),
                    evidence_level=normalize_evidence_level(" ".join(pub_types), title), journal=journal, doi=doi,
                    citation_eligible=True, record_type="published_evidence", source_locator=f"PMID:{pmid}",
                    provenance=provenance, extra={"pub_types": pub_types}))
            time.sleep(0.34)
    return docs

def _tags_from_text(text: str) -> list[str]:
    mapping = {
        "hypertension": ["hypertension", "blood pressure", "高血压", "血压"],
        "hyperlipidemia": ["lipid", "cholesterol", "血脂", "胆固醇", "LDL", "HDL"],
        "diabetes": ["diabetes", "glycemic", "糖尿病", "血糖"],
        "cardiovascular": ["cardiovascular", "coronary", "心血管", "冠心病"],
        "diet": ["diet", "dietary", "nutrition", "饮食", "营养", "sodium", "钠"],
        "mediterranean": ["mediterranean", "地中海"],
        "guideline": ["guideline", "指南", "recommendation"],
    }
    lower = text.lower()
    tags = []
    for tag, keys in mapping.items():
        if any(k.lower() in lower for k in keys):
            tags.append(tag)
    return tags


# 默认检索任务：按 PICO（人群/干预/结局）结构化生成，见 build_mesh_aware_query
DEFAULT_PICO = [
    {"disease": "hypertension", "intervention": "antihypertensive agents", "outcome": "cardiovascular events"},
    {"disease": "hyperlipidemia", "intervention": "lifestyle intervention OR statin", "outcome": "LDL cholesterol"},
    {"disease": "mediterranean diet", "intervention": "", "outcome": "cardiovascular risk"},
    {"disease": "hypertension", "intervention": "sodium reduction OR dietary sodium", "outcome": "blood pressure"},
    {"disease": "type 2 diabetes", "intervention": "dietary intervention", "outcome": "cardiovascular risk"},
]


def ingest_pubmed(queries: list[str] | None = None, retmax_per_query: int = 15) -> list[EvidenceDoc]:
    """Batch-ingest PubMed records while retaining searchable raw responses."""
    queries = queries or [build_mesh_aware_query(**p) for p in DEFAULT_PICO]
    try:
        run = start_ingest_run("pubmed", queries, {"retmax_per_query": retmax_per_query})
    except Exception as exc:
        logger.warning("PubMed run initialization failed: %s", exc)
        return []
    docs: list[EvidenceDoc] = []
    all_ids: list[str] = []
    pmid_queries: dict[str, list[str]] = {}
    setattr(run, "pmid_queries", pmid_queries)
    try:
        for query in queries:
            try:
                ids = search_pmids(query, retmax=retmax_per_query, run=run)
            except Exception as exc:
                message = f"source=pubmed query={query!r} {type(exc).__name__}: {exc}"
                run.errors.append(message)
                logger.warning("PubMed search failed: %s", message)
                continue
            logger.info("PubMed query=%r -> %d ids", query, len(ids))
            for pmid in ids:
                all_ids.append(pmid)
                matched_queries = pmid_queries.setdefault(pmid, [])
                if query not in matched_queries:
                    matched_queries.append(query)
            time.sleep(0.34)
        seen: set[str] = set()
        unique_pmids = []
        for pmid in all_ids:
            if pmid not in seen:
                seen.add(pmid)
                unique_pmids.append(pmid)
        docs.extend(fetch_pubmed_docs(unique_pmids, run=run))
    except Exception as exc:
        message = f"source=pubmed {type(exc).__name__}: {exc}"
        run.errors.append(message)
        logger.warning("PubMed live fetch failed: %s", message)
    finally:
        run.finalize(docs)
    return docs

# ---------------------------------------------------------------------------
# PubMed 采集增强
# ---------------------------------------------------------------------------


def _pico_term(term: str, *, mesh: bool = False) -> str:
    """把 PICO 单个要素转成 PubMed 检索片段：多词加引号，支持 OR 拆分，可挂 MeSH。"""
    subs = [s for s in re.split(r"\s+OR\s+", term.strip(), flags=re.IGNORECASE) if s.strip()]
    parts = []
    for sub in subs:
        sub = " ".join(sub.split())
        quoted = f'"{sub}"' if len(sub.split()) > 1 else sub
        if mesh:
            parts.append(f"{quoted}[Title/Abstract] OR {sub}[MeSH Terms]")
        else:
            parts.append(f"{quoted}[Title/Abstract]")
    return f"({ ' OR '.join(parts) })" if len(parts) > 1 else parts[0]


def build_mesh_aware_query(disease: str, intervention: str = "", outcome: str = "") -> str:
    """
    根据疾病/干预/结局组装 PICO 结构化 PubMed 检索式（可含 MeSH）。

    创新点：
        - 按 PICO 分段用 AND 组合，避免整句模糊匹配，召回/精度可分别调优；
        - 疾病要素额外挂 MeSH Terms 提升召回，干预/结局限定 Title/Abstract 提升精度；
        - 要素内部支持 "A OR B" 拆分，保留 OR 语义；
        - 末尾统一追加 hasabstract[text]，过滤无摘要记录，保证正文可用。

    参数:
        disease: 疾病或人群关键词。
        intervention: 干预措施（可空）。
        outcome: 结局指标（可空）。

    返回:
        str: 可直接传给 search_pmids 的检索式。

    作用:
        提高文献召回精度，减少噪声 PMID。
    """
    parts = []
    if disease:
        parts.append(_pico_term(disease, mesh=True))
    if intervention:
        parts.append(_pico_term(intervention))
    if outcome:
        parts.append(_pico_term(outcome))
    if not parts:
        return ""
    return " AND ".join(parts) + " AND hasabstract[text]"
