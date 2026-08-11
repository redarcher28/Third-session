# -*- coding: utf-8 -*-
"""
ClinicalTrials.gov 采集模块（Data API v2）。

职责：按病种/条件检索临床试验，映射为统一 EvidenceDoc。
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.models import EvidenceDoc

logger = logging.getLogger(__name__)

API = "https://clinicaltrials.gov/api/v2/studies"


def _tags(text: str) -> list[str]:
    mapping = {
        "hypertension": ["hypertension", "blood pressure"],
        "hyperlipidemia": ["hyperlipid", "cholesterol", "dyslipid"],
        "diabetes": ["diabetes"],
        "cardiovascular": ["cardiovascular", "coronary"],
        "diet": ["diet", "nutrition", "sodium"],
    }
    lower = text.lower()
    return [t for t, keys in mapping.items() if any(k in lower for k in keys)]


def search_trials(condition: str, page_size: int = 20) -> list[EvidenceDoc]:
    """
    按 condition 检索临床试验记录。

    参数:
        condition: 病种或条件关键词（英文为主，如 Hypertension）。
        page_size: 返回条数。

    返回:
        list[EvidenceDoc]: 试验摘要文档（doc_id 形如 nct:NCTxxxxxxxx）。
    """
    params = {
        "query.cond": condition,
        "pageSize": page_size,
        "format": "json",
    }
    with httpx.Client(timeout=60) as client:
        r = client.get(API, params=params)
        r.raise_for_status()
        payload = r.json()
    docs: list[EvidenceDoc] = []
    for study in payload.get("studies", []):
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        nct = ident.get("nctId", "")
        title = ident.get("officialTitle") or ident.get("briefTitle") or nct
        desc = proto.get("descriptionModule", {})
        brief = desc.get("briefSummary") or desc.get("detailedDescription") or title
        status = proto.get("statusModule", {}).get("overallStatus", "")
        conditions = proto.get("conditionsModule", {}).get("conditions", []) or []
        interventions = []
        for arm in proto.get("armsInterventionsModule", {}).get("interventions", []) or []:
            name = arm.get("name") or ""
            if name:
                interventions.append(name)
        primary_outcome = extract_trial_primary_outcome(study)
        year = None
        start = proto.get("statusModule", {}).get("startDateStruct", {}).get("date")
        if start and len(start) >= 4 and start[:4].isdigit():
            year = int(start[:4])
        text = (
            f"{brief}\n\nStatus: {status}\n"
            f"Conditions: {', '.join(conditions)}\n"
            f"Interventions: {', '.join(interventions)}\n"
            f"Primary outcomes: {primary_outcome}"
        )
        blob = f"{title} {text}"
        docs.append(
            EvidenceDoc(
                doc_id=f"nct:{nct}",
                source="clinicaltrials",
                title=title,
                text=text[:8000],
                year=year,
                url=f"https://clinicaltrials.gov/study/{nct}",
                tags=_tags(blob),
                evidence_level="rct",
                extra={
                    "status": status,
                    "conditions": conditions,
                    "interventions": interventions,
                    "primary_outcome": primary_outcome,
                },
            )
        )
    return docs


DEFAULT_CONDITIONS = [
    "Hypertension",
    "Hyperlipidemia",
    "Type 2 Diabetes Mellitus",
    "Mediterranean Diet",
]


def ingest_clinicaltrials(
    conditions: list[str] | None = None,
    page_size: int = 10,
) -> list[EvidenceDoc]:
    """
    批量按多个病种采集临床试验。

    参数:
        conditions: 病种列表；None 使用默认列表。
        page_size: 每个病种拉取条数。

    返回:
        list[EvidenceDoc]: 合并后的试验文档列表。
    """
    conditions = conditions or DEFAULT_CONDITIONS
    out: list[EvidenceDoc] = []
    try:
        for cond in conditions:
            docs = search_trials(cond, page_size=page_size)
            logger.info("ClinicalTrials condition=%r -> %d", cond, len(docs))
            out.extend(docs)
    except Exception as e:
        logger.warning("ClinicalTrials fetch failed: %s", e)
        return []
    return out


# ---------------------------------------------------------------------------
# 【待完善】临床试验字段增强（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def extract_trial_primary_outcome(study_json: dict) -> str:
    """
    【待完善】从 ClinicalTrials API 原始 JSON 中提取主要结局指标描述。

    参数:
        study_json: API 返回的单条 study 对象。

    返回:
        str: 主要结局文本；缺失则返回空字符串。

    作用:
        丰富试验证据正文，提升回答时对「结局」相关问题的命中率。
    """
    proto = study_json.get("protocolSection", {}) if isinstance(study_json, dict) else {}
    outcomes = proto.get("outcomesModule", {}).get("primaryOutcomes", []) or []
    lines: list[str] = []
    for item in outcomes:
        if not isinstance(item, dict):
            continue
        measure = item.get("measure") or item.get("title") or ""
        description = item.get("description") or ""
        time_frame = item.get("timeFrame") or ""
        parts = [str(x).strip() for x in [measure, description, time_frame] if str(x).strip()]
        if parts:
            lines.append(" | ".join(parts))
    return "; ".join(lines)
