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
from src.ingest import normalize_evidence_level, normalize_evidence_metadata
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
            try:
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
            except Exception as e:
                # 单个 efetch 批次失败时跳过该批，避免整批 PubMed 文献被丢弃。
                logger.warning("PubMed efetch batch failed ids=%s error=%s", ",".join(batch[:3]), e)
                continue
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
                    normalize_evidence_metadata(
                        EvidenceDoc(
                            doc_id=f"pmid:{pmid}",
                            source="pubmed",
                            title=title or f"PMID {pmid}",
                            text=abstract,
                            year=year,
                            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            tags=tags,
                            evidence_level=normalize_evidence_level(" ".join(pub_types), title),
                            journal=journal,
                            doi=doi,
                            record_type="published_article",
                            citation_eligible=True,
                            source_locator=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            extra={"pub_types": pub_types},
                        )
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
        "dash": ["dash"],
        "fiber": ["fiber", "fibre", "whole grain", "whole-grain", "膳食纤维", "全谷物"],
        "plant_based": ["plant-based", "plant based", "vegetarian", "植物性", "素食"],
        "sugar": ["sugar-sweetened", "sugary drink", "added sugar", "含糖"],
        "ultra_processed": ["ultra-processed", "ultraprocessed", "超加工"],
        "obesity": ["obesity", "overweight", "weight loss", "体重", "肥胖"],
        "low_carb": ["low-carbohydrate", "low carbohydrate", "低碳"],
        "omega3": ["omega-3", "omega 3"],
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


def ingest_pubmed(
    queries: list[str] | None = None,
    retmax_per_query: int = 15,
) -> list[EvidenceDoc]:
    """
    按默认/自定义查询批量采集 PubMed 文献。

    参数:
        queries: 检索式列表；None 时由 DEFAULT_PICO 用 build_mesh_aware_query 生成。
        retmax_per_query: 每个查询最多拉取条数。

    返回:
        list[EvidenceDoc]: 去重前的原始采集结果（上层可再 merge）。
    """
    queries = queries or [build_mesh_aware_query(**p) for p in DEFAULT_PICO]
    all_ids: list[str] = []
    for q in queries:
        try:
            ids = search_pmids(q, retmax=retmax_per_query)
            logger.info("PubMed query=%r -> %d ids", q, len(ids))
            all_ids.extend(ids)
            time.sleep(0.34)
        except Exception as e:
            # 单条检索失败不应丢掉整批已成功的 PMID，网络抖动时可继续保留权威文献来源。
            logger.warning("PubMed query failed query=%r error=%s", q, e)
            continue
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
