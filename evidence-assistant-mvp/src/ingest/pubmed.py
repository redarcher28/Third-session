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
from src.ingest import normalize_evidence_level
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


def search_pmids(query: str, retmax: int = 20) -> list[str]:
    """
    使用 esearch 按关键词检索 PMID 列表。

    参数:
        query: PubMed 检索式。
        retmax: 最多返回条数。

    返回:
        list[str]: PMID 字符串列表。
    """
    with httpx.Client(timeout=60) as client:
        r = client.get(
            f"{EUTILS}/esearch.fcgi",
            params=_params(db="pubmed", term=query, retmax=retmax, retmode="json"),
        )
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        return list(ids)


def _text(el: ET.Element | None, path: str = "") -> str:
    if el is None:
        return ""
    node = el.find(path) if path else el
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _infer_level(title: str, pub_types: list[str]) -> str:
    return normalize_evidence_level(" ".join(pub_types), title)


def fetch_pubmed_docs(pmids: list[str]) -> list[EvidenceDoc]:
    """
    使用 efetch 批量拉取文献摘要并转为 EvidenceDoc。

    参数:
        pmids: PMID 列表。

    返回:
        list[EvidenceDoc]: 含标题、摘要、年份、链接等字段的文档。
    """
    if not pmids:
        return []
    docs: list[EvidenceDoc] = []
    # Batch efetch
    with httpx.Client(timeout=90) as client:
        for i in range(0, len(pmids), 50):
            batch = pmids[i : i + 50]
            r = client.get(
                f"{EUTILS}/efetch.fcgi",
                params=_params(
                    db="pubmed",
                    id=",".join(batch),
                    retmode="xml",
                ),
            )
            r.raise_for_status()
            root = ET.fromstring(r.text)
            for article in root.findall(".//PubmedArticle"):
                pmid = _text(article, ".//PMID")
                title = _text(article, ".//ArticleTitle")
                abstract_parts = [
                    "".join(n.itertext()).strip()
                    for n in article.findall(".//Abstract/AbstractText")
                ]
                abstract = "\n".join(p for p in abstract_parts if p)
                year_txt = _text(article, ".//PubDate/Year") or _text(
                    article, ".//PubDate/MedlineDate"
                )[:4]
                year = int(year_txt) if year_txt.isdigit() else None
                journal = _text(article, ".//Journal/Title")
                doi = ""
                for aid in article.findall(".//ArticleId"):
                    if aid.get("IdType") == "doi":
                        doi = (aid.text or "").strip()
                pub_types = [
                    "".join(pt.itertext()).strip()
                    for pt in article.findall(".//PublicationType")
                ]
                if not abstract:
                    abstract = title
                tags = _tags_from_text(f"{title} {abstract}")
                docs.append(
                    EvidenceDoc(
                        doc_id=f"pmid:{pmid}",
                        source="pubmed",
                        title=title or f"PMID {pmid}",
                        text=abstract,
                        year=year,
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        tags=tags,
                        evidence_level=_infer_level(title, pub_types),  # type: ignore[arg-type]
                        journal=journal,
                        doi=doi,
                        extra={"pub_types": pub_types},
                    )
                )
            time.sleep(0.34)  # be polite to NCBI
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


DEFAULT_QUERIES = [
    "hypertension long-term antihypertensive therapy guidelines[Publication Type]",
    "hyperlipidemia lifestyle intervention OR statin evidence",
    "Mediterranean diet cardiovascular risk meta-analysis",
    "sodium reduction hypertension systematic review",
    "diabetes dietary intervention cardiovascular",
]


def ingest_pubmed(
    queries: list[str] | None = None,
    retmax_per_query: int = 15,
) -> list[EvidenceDoc]:
    """
    按默认/自定义查询批量采集 PubMed 文献。

    参数:
        queries: 检索式列表；None 时使用 DEFAULT_QUERIES。
        retmax_per_query: 每个查询最多拉取条数。

    返回:
        list[EvidenceDoc]: 去重前的原始采集结果（上层可再 merge）。
    """
    queries = queries or DEFAULT_QUERIES
    all_ids: list[str] = []
    try:
        for q in queries:
            ids = search_pmids(q, retmax=retmax_per_query)
            logger.info("PubMed query=%r -> %d ids", q, len(ids))
            all_ids.extend(ids)
            time.sleep(0.34)
    except Exception as e:
        logger.warning("PubMed live fetch failed: %s — using empty list", e)
        return []
    # unique preserve order
    seen: set[str] = set()
    uniq = []
    for i in all_ids:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    try:
        return fetch_pubmed_docs(uniq)
    except Exception as e:
        logger.warning("PubMed efetch failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# 【待完善】PubMed 采集增强（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def build_mesh_aware_query(disease: str, intervention: str = "", outcome: str = "") -> str:
    """
    【待完善】根据疾病/干预/结局组装更规范的 PubMed 检索式（可含 MeSH）。

    参数:
        disease: 疾病或人群关键词。
        intervention: 干预措施（可空）。
        outcome: 结局指标（可空）。

    返回:
        str: 可直接传给 search_pmids 的检索式。

    作用:
        提高文献召回精度，减少噪声 PMID。
    """
    def clean(value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip())

    def term_block(value: str, mesh: str | None = None) -> str:
        value = clean(value)
        if not value:
            return ""
        escaped = value.replace('"', "")
        pieces = [f'"{escaped}"[Title/Abstract]']
        if mesh:
            pieces.insert(0, f'"{mesh}"[MeSH Terms]')
        return "(" + " OR ".join(pieces) + ")"

    mesh_map = {
        "hypertension": "Hypertension",
        "high blood pressure": "Hypertension",
        "高血压": "Hypertension",
        "hyperlipidemia": "Hyperlipidemias",
        "dyslipidemia": "Dyslipidemias",
        "cholesterol": "Cholesterol",
        "血脂": "Hyperlipidemias",
        "diabetes": "Diabetes Mellitus, Type 2",
        "type 2 diabetes": "Diabetes Mellitus, Type 2",
        "糖尿病": "Diabetes Mellitus, Type 2",
        "mediterranean diet": "Diet, Mediterranean",
        "地中海饮食": "Diet, Mediterranean",
        "dash diet": "Diet, Sodium-Restricted",
        "sodium": "Sodium, Dietary",
        "salt": "Sodium, Dietary",
        "钠": "Sodium, Dietary",
        "盐": "Sodium, Dietary",
        "statin": "Hydroxymethylglutaryl-CoA Reductase Inhibitors",
        "他汀": "Hydroxymethylglutaryl-CoA Reductase Inhibitors",
        "cardiovascular": "Cardiovascular Diseases",
        "心血管": "Cardiovascular Diseases",
    }

    disease_clean = clean(disease)
    if not disease_clean:
        return ""
    disease_mesh = mesh_map.get(disease_clean.lower())
    parts = [term_block(disease_clean, disease_mesh)]

    intervention_clean = clean(intervention)
    if intervention_clean:
        parts.append(term_block(intervention_clean, mesh_map.get(intervention_clean.lower())))

    outcome_clean = clean(outcome)
    if outcome_clean:
        parts.append(term_block(outcome_clean, mesh_map.get(outcome_clean.lower())))

    evidence_filter = (
        "(guideline[Publication Type] OR practice guideline[Publication Type] OR "
        "meta-analysis[Publication Type] OR systematic review[Title/Abstract] OR "
        "randomized controlled trial[Publication Type] OR clinical trial[Publication Type])"
    )
    return " AND ".join([p for p in parts if p] + [evidence_filter])
