# -*- coding: utf-8 -*-
"""
证据助手 Tool 层：给模型可调用的结构化外部能力（D2 材料：Tool/MCP）。

设计红线（对应材料第 10 页）：
    - 一 Tool 一事，不做 medical_search_everything；
    - 输入输出优先 JSON，返回可读的结构化错误，不吞异常；
    - 防呆与限权：retmax 封顶、不打印任何密钥；
    - 故障预案：网络失败时返回缓存 JSON（材料：演示超时/失效时展示缓存）。

三个 Tool：
    search_pubmed    检索式 → 标题/PMID/摘要/链接（补本地库时效）
    verify_citation  PMID/DOI → 是否存在/元数据/撤稿提示（核验引用）
    get_trial_record  NCT ID → 试验状态/分期/干预（回查注册试验）
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from src.config import get_settings
from src.ingest.pubmed import _params, fetch_pubmed_docs, search_pmids

logger = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_RETMAX = 50  # 防呆：单次检索条数上限
_CACHE_FILE = "data/cache/tool_cache.json"


# ---------------------------------------------------------------------------
# 缓存与结构化错误（材料：网络/MCP 失效时展示预先缓存的 JSON）
# ---------------------------------------------------------------------------


def _load_cache() -> dict[str, Any]:
    p = Path(_CACHE_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    Path(_CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(_CACHE_FILE).write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _err(msg: str, tool: str, **extra: Any) -> dict[str, Any]:
    """结构化错误返回（材料：失败返回可读的结构化错误，不吞异常）。"""
    return {"ok": False, "tool": tool, "error": msg, **extra}


def _retry_call(fn, times: int = 3, delay: float = 1.5):
    """
    连接类错误自动重试（代理/网络抖动时演示更稳；材料：演示超时停在第③步，
    重试耗尽后才进入缓存/错误路径）。
    """
    last: Exception | None = None
    for i in range(times):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < times - 1:
                time.sleep(delay * (i + 1))
    raise last  # type: ignore[misc]


def _cached_or_err(tool: str, key: str, error: str) -> dict[str, Any]:
    """网络失败时优先返回缓存；无缓存则返回结构化错误 + 使用建议。"""
    cache = _load_cache()
    hit = cache.get(tool, {}).get(key)
    if hit:
        return {"ok": True, "cached": True, "stale_note": "网络不可用，返回上次成功结果（缓存）", **hit}
    return _err(error, tool, hint="网络不可用。可配置离线模式演示，或稍后重试。")


def _cache_put(tool: str, key: str, payload: dict[str, Any]) -> None:
    cache = _load_cache()
    cache.setdefault(tool, {})[key] = payload
    _save_cache(cache)


# ---------------------------------------------------------------------------
# Tool 1: search_pubmed —— 检索式 → 标题/PMID/摘要/链接
# ---------------------------------------------------------------------------


def search_pubmed(
    query: str,
    *,
    year_from: int | None = None,
    year_to: int | None = None,
    article_type: str | None = None,
    retmax: int = 12,
) -> dict[str, Any]:
    """
    按检索式查 PubMed，返回结构化文献列表。

    参数:
        query: 检索式（可含限钠/血压等关键词或已有 [Title/Abstract] 语法）。
        year_from: 起始年（含）；year_to: 结束年（含）。
        article_type: 文献类型（如 systematic review / guideline / randomized controlled trial）。
        retmax: 最多返回条数（防呆上限 50）。

    返回:
        {"ok": true, "items": [{pmid, title, year, journal, url, abstract}], "count": n}
        失败时返回 {"ok": false, "error": ..., "hint": ...}
    """
    tool = "search_pubmed"
    key = json.dumps([query, year_from, year_to, article_type, retmax], ensure_ascii=False)
    retmax = max(1, min(int(retmax), MAX_RETMAX))
    try:
        term = query.strip()
        if year_from or year_to:
            term += f" AND {year_from or 1800}:{year_to or 3000}[dp]"
        if article_type:
            term += f' AND "{article_type}"[pt]'
        pmids = _retry_call(lambda: search_pmids(term, retmax=retmax))
        if not pmids:
            return {"ok": True, "items": [], "count": 0, "note": "无匹配文献"}
        docs = _retry_call(lambda: fetch_pubmed_docs(pmids))
        if not docs:
            # fetch_pubmed_docs 内部吞异常返回空列表，无法区分「真无摘要」与「网络失败」，
            # 按失败处理以触发重试/缓存兜底（材料：演示超时停在第③步，不硬答）
            raise ConnectionError("PubMed 摘要拉取返回空，疑似网络失败")
        items = [
            {
                "pmid": d.doc_id.replace("pmid:", ""),
                "title": d.title,
                "year": d.year,
                "journal": d.journal,
                "url": d.url,
                "evidence_level": d.evidence_level,
                "abstract": (d.text or "")[:200],
            }
            for d in docs
        ]
        payload = {"items": items, "count": len(items)}
        _cache_put(tool, key, payload)
        return {"ok": True, **payload}
    except Exception as e:
        logger.warning("search_pubmed failed: %s", e)
        return _cached_or_err(tool, key, f"PubMed 检索失败: {e}")


# ---------------------------------------------------------------------------
# Tool 2: verify_citation —— PMID/DOI → 是否存在/元数据/撤稿提示
# ---------------------------------------------------------------------------


def verify_citation(identifier: str) -> dict[str, Any]:
    """
    核验文献标识（PMID 或 DOI）是否存在、取元数据、提示撤稿。

    参数:
        identifier: "pmid:12345678" / "12345678" / "10.xxxx/yyyy"。

    返回:
        {"ok": true, "exists": bool, "metadata": {pmid,title,year,journal,doi,url},
         "withdrawn": bool, "note": str}
    """
    tool = "verify_citation"
    key = identifier.strip().lower()
    ident = key.replace("pmid:", "").strip()
    is_doi = "/" in ident and ident.lower().startswith("10.")
    try:
        with httpx.Client(timeout=60) as client:
            if is_doi:
                term = f'"{ident}"[AID]'
            else:
                term = f"{ident}[UID]"
            r = _retry_call(lambda: client.get(
                f"{EUTILS}/esearch.fcgi",
                params=_params(db="pubmed", term=term, retmax=5, retmode="json"),
            ))
            r.raise_for_status()
            idlist = r.json().get("esearchresult", {}).get("idlist", [])
            if not idlist:
                payload = {"exists": False, "metadata": {}, "withdrawn": False,
                           "note": f"未找到对应记录：{identifier}（可能为编造或录入错误）"}
                return {"ok": True, **payload}
            # efetch 元数据 + 撤稿检查
            pmid = idlist[0]
            r2 = _retry_call(lambda: client.get(
                f"{EUTILS}/efetch.fcgi",
                params=_params(db="pubmed", id=pmid, retmode="xml"),
            ))
            r2.raise_for_status()
            xml = r2.text
            import re

            title = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", xml, re.S)
            year = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", xml, re.S)
            journal = re.search(r"<Title>(.*?)</Title>", xml, re.S)
            withdrawn = "Retraction" in xml or "Retracted" in xml
            payload = {
                "exists": True,
                "metadata": {
                    "pmid": pmid,
                    "title": (title.group(1).strip() if title else ""),
                    "year": int(year.group(1)) if year else None,
                    "journal": (journal.group(1) if journal else ""),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                },
                "withdrawn": withdrawn,
                "note": "该文献已被撤稿/标记撤回，引用前需人工复核" if withdrawn else "文献存在，可回查",
            }
        _cache_put(tool, key, payload)
        return {"ok": True, **payload}
    except Exception as e:
        logger.warning("verify_citation failed: %s", e)
        return _cached_or_err(tool, key, f"文献核验失败: {e}")


# ---------------------------------------------------------------------------
# Tool 3: get_trial_record —— NCT ID → 试验状态/分期/干预
# ---------------------------------------------------------------------------


def get_trial_record(nct_id: str) -> dict[str, Any]:
    """
    按注册号回查临床试验：状态/分期/干预/条件。

    参数:
        nct_id: 形如 NCT01234567。

    返回:
        {"ok": true, "metadata": {nct_id,title,status,phase,enrollment,
         conditions,interventions,url}}
    """
    tool = "get_trial_record"
    nct = nct_id.strip().upper()
    key = nct
    if not nct.startswith("NCT") or not nct[3:].isdigit():
        return _err("注册号格式应为 NCT 加数字，如 NCT01234567", tool)
    try:
        from src.ingest.clinicaltrials import API

        with httpx.Client(timeout=60) as client:
            r = _retry_call(lambda: client.get(
                API, params={"query.term": nct, "pageSize": 5, "format": "json"}
            ))
            r.raise_for_status()
            studies = r.json().get("studies", [])
        if not studies:
            return {"ok": True, "metadata": {}, "note": f"未找到注册试验：{nct}"}
        proto = studies[0].get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        arms = proto.get("armsInterventionsModule", {}) or {}
        interventions = [
            a.get("name") or "" for a in arms.get("interventions", []) if a.get("name")
        ]
        payload = {
            "metadata": {
                "nct_id": ident.get("nctId", nct),
                "title": ident.get("officialTitle") or ident.get("briefTitle") or nct,
                "status": status.get("overallStatus", ""),
                "phase": design.get("phases", []),
                "enrollment": (design.get("enrollmentInfo", {}) or {}).get("count"),
                "conditions": proto.get("conditionsModule", {}).get("conditions", []),
                "interventions": interventions,
                "url": f"https://clinicaltrials.gov/study/{nct}",
            },
            "note": "注册试验可回查",
        }
        _cache_put(tool, key, payload)
        return {"ok": True, **payload}
    except Exception as e:
        logger.warning("get_trial_record failed: %s", e)
        return _cached_or_err(tool, key, f"试验记录查询失败: {e}")


# ---------------------------------------------------------------------------
# 证据卡：把核验通过的元数据格式化为展示用卡片（材料：生成证据卡）
# ---------------------------------------------------------------------------


def format_evidence_card(metadata: dict[str, Any]) -> str:
    """把核验通过的文献元数据格式化为证据卡文本（标题|年份|类型|链接）。"""
    m = metadata or {}
    return (
        f"标题：{m.get('title') or '（无标题）'}\n"
        f"年份：{m.get('year') or '未知'} | 期刊：{m.get('journal') or '未知'}\n"
        f"链接：{m.get('url') or ''}"
    )


if __name__ == "__main__":  # 冒烟自检
    r = search_pubmed("sodium reduction hypertension", year_from=2018, retmax=3)
    print(json.dumps(r, ensure_ascii=False)[:400])
