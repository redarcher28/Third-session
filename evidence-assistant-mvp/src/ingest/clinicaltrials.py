# -*- coding: utf-8 -*-
"""
ClinicalTrials.gov 采集模块（Data API v2）。

职责：按病种/条件检索临床试验，映射为统一 EvidenceDoc。
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.ingest import IngestRun, start_ingest_run
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


def _extract_trial_results_status(study: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return only a results status supported by fields present in this API record."""
    if "hasResults" in study:
        value = study.get("hasResults")
        if isinstance(value, bool):
            return ("available" if value else "not_available"), {"path": "hasResults", "value": value}
        if value is not None:
            return str(value), {"path": "hasResults", "value": value}

    status_module = (study.get("protocolSection") or {}).get("statusModule") or {}
    for field, label in (("resultsFirstPostDateStruct", "posted"), ("resultsFirstPostDate", "posted"), ("resultsFirstSubmitDate", "submitted")):
        if field in status_module and status_module.get(field) is not None:
            return label, {"path": f"protocolSection.statusModule.{field}", "value": status_module.get(field)}

    if "resultsSection" in study and study.get("resultsSection") is not None:
        return "available", {"path": "resultsSection", "value": study.get("resultsSection")}
    return "", {}


def search_trials(condition: str, page_size: int = 20, *, run: IngestRun | None = None) -> list[EvidenceDoc]:
    """Search ClinicalTrials.gov and optionally retain the successful JSON response."""
    params = {"query.cond": condition, "pageSize": page_size, "format": "json"}
    with httpx.Client(timeout=60) as client:
        response = client.get(API, params=params)
        response.raise_for_status()
        raw_path = run.save_response(response.content, "json") if run else ""
        payload = response.json()

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
        interventions = [
            intervention.get("name") or ""
            for intervention in proto.get("armsInterventionsModule", {}).get("interventions", []) or []
            if intervention.get("name")
        ]
        year = None
        start = proto.get("statusModule", {}).get("startDateStruct", {}).get("date")
        if start and len(start) >= 4 and start[:4].isdigit():
            year = int(start[:4])
        primary_outcome = extract_trial_primary_outcome(study)
        results_status, results_status_source = _extract_trial_results_status(study)
        text = f"{brief}\n\nStatus: {status}\nConditions: {', '.join(conditions)}\nInterventions: {', '.join(interventions)}"
        if primary_outcome:
            text += f"\nPrimary Outcome: {primary_outcome}"
        provenance = {}
        if run is not None:
            provenance = {
                "run_id": run.run_id,
                "source": "clinicaltrials",
                "query": condition,
                "nct": nct,
                "overall_status": status,
                "interventions": interventions,
                "primary_outcome": primary_outcome,
                "results_status": results_status,
                "results_status_source": results_status_source,
                "retrieved_at": run.started_at,
                "raw_response_path": raw_path,
            }
        docs.append(EvidenceDoc(
            doc_id=f"nct:{nct}", source="clinicaltrials", title=title, text=text[:8000], year=year,
            url=f"https://clinicaltrials.gov/study/{nct}", tags=_tags(f"{title} {text}"),
            evidence_level="other", citation_eligible=True, record_type="trial_registration",
            source_locator=f"NCT:{nct}", provenance=provenance,
            extra={"status": status, "conditions": conditions, "primary_outcome": primary_outcome,
                   "results_status": results_status, "results_status_source": results_status_source},
        ))
    return docs

DEFAULT_CONDITIONS = [
    "Hypertension",
    "Hyperlipidemia",
    "Type 2 Diabetes Mellitus",
    "Mediterranean Diet",
]


def ingest_clinicaltrials(conditions: list[str] | None = None, page_size: int = 10) -> list[EvidenceDoc]:
    """Batch-ingest trial registrations while retaining raw query responses."""
    conditions = conditions or DEFAULT_CONDITIONS
    try:
        run = start_ingest_run("clinicaltrials", conditions, {"page_size": page_size})
    except Exception as exc:
        logger.warning("ClinicalTrials run initialization failed: %s", exc)
        return []

    docs: list[EvidenceDoc] = []
    try:
        for condition in conditions:
            try:
                condition_docs = search_trials(condition, page_size=page_size, run=run)
            except Exception as exc:
                message = f"source=clinicaltrials query={condition!r} {type(exc).__name__}: {exc}"
                run.errors.append(message)
                logger.warning("ClinicalTrials query failed: %s", message)
                continue
            logger.info("ClinicalTrials condition=%r -> %d", condition, len(condition_docs))
            docs.extend(condition_docs)
    except Exception as exc:
        message = f"source=clinicaltrials {type(exc).__name__}: {exc}"
        run.errors.append(message)
        logger.warning("ClinicalTrials fetch failed: %s", message)
    finally:
        run.finalize(docs)
    return docs

# ---------------------------------------------------------------------------
# 临床试验字段增强
# ---------------------------------------------------------------------------


def extract_trial_primary_outcome(study_json: dict) -> str:
    """
    从 ClinicalTrials API v2 原始 JSON 中提取主要结局指标描述。

    创新点：
        - 输出「指标（时间窗）」结构化文本，缺时间窗时仅保留指标；
        - 主要结局缺失时用次要结局兜底，保证结局字段尽量非空；
        - 结果同时写入 extra["primary_outcome"]，供 B 组检索过滤/展示使用。

    参数:
        study_json: API 返回的单条 study 对象。

    返回:
        str: 主要结局文本；缺失则返回空字符串。

    作用:
        丰富试验证据正文，提升回答时对「结局」相关问题的命中率。
    """
    proto = (study_json or {}).get("protocolSection", {})
    outcomes = proto.get("outcomesModule", {}) or {}
    for key in ("primaryOutcomes", "secondaryOutcomes"):
        blocks = []
        for o in outcomes.get(key) or []:
            measure = (o.get("measure") or "").strip()
            if not measure:
                continue
            tf = (o.get("timeFrame") or "").strip()
            blocks.append(measure + (f"（{tf}）" if tf else ""))
        if blocks:
            return "；".join(blocks)[:500]
    return ""
