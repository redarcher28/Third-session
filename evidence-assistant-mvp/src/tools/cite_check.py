# -*- coding: utf-8 -*-
"""
引用校验工具：检测假引用、无效编号、编造 PMID/NCT。
"""

from __future__ import annotations

import re
from typing import Any

from src.generation.answer import DISCLAIMER, extract_citation_indices


PMID_RE = re.compile(r"\bPMID[:\s]*([0-9]{5,9})\b", re.I)
NCT_RE = re.compile(r"\b(NCT\d{8})\b", re.I)
DOC_RE = re.compile(r"\b((?:pmid|nct|epmc|local|wiki):[^\s\]，。；;,]+)", re.I)
REF_SECTION_MARKERS = ("**参考文献**", "\n参考文献")


def split_answer_body(answer: str) -> tuple[str, str]:
    """拆分模型正文与参考文献块。"""
    for marker in REF_SECTION_MARKERS:
        idx = answer.find(marker)
        if idx >= 0:
            start = answer.rfind("\n---", 0, idx)
            cut = start if start >= 0 and idx - start < 40 else idx
            return answer[:cut].rstrip(), answer[cut:].lstrip()
    return answer, ""


def answer_body_for_citation_check(answer: str) -> str:
    """用于引用校验的正文（不含自动参考文献与免责声明）。"""
    body, _ = split_answer_body(answer)
    disclaimer = DISCLAIMER.strip()
    if disclaimer and disclaimer in body:
        body = body[: body.rfind(disclaimer)].rstrip()
    return body


def verify_citations(
    answer: str,
    contexts: list[dict[str, Any]],
    *,
    body_only: bool = True,
) -> dict[str, Any]:
    """
    校验回答中的引用是否都能在当次证据中找到。

    默认只扫描模型正文，不把自动追加的「参考文献」段落算入。
    """
    text = answer_body_for_citation_check(answer) if body_only else answer
    allowed_idx = set(range(1, len(contexts) + 1))
    used = extract_citation_indices(text)
    invalid_brackets = [i for i in used if i not in allowed_idx]

    context_ids: set[str] = set()
    context_pmids: set[str] = set()
    context_ncts: set[str] = set()
    for c in contexts:
        doc_id = str(c.get("doc_id") or "")
        context_ids.add(doc_id.lower())
        if doc_id.lower().startswith("pmid:"):
            context_pmids.add(doc_id.split(":", 1)[1])
        if doc_id.lower().startswith("nct:"):
            context_ncts.add(doc_id.split(":", 1)[1].upper())
        for m in PMID_RE.findall(str(c.get("url") or "") + " " + doc_id):
            context_pmids.add(m)
        for m in NCT_RE.findall(str(c.get("url") or "") + " " + doc_id):
            context_ncts.add(m.upper())

    claimed_pmids = PMID_RE.findall(text)
    claimed_ncts = [m.upper() for m in NCT_RE.findall(text)]
    claimed_docs = [m.lower() for m in DOC_RE.findall(text)]

    fake_pmids = [p for p in claimed_pmids if p not in context_pmids]
    fake_ncts = [n for n in claimed_ncts if n not in context_ncts]
    fake_docs = [d for d in claimed_docs if d not in context_ids]

    invent_phrases = 0
    for pat in [
        r"根据某项研究",
        r"著名的 .+ 试验",
        r"发表于《[^》]+》",
        r"et al\.",
    ]:
        invent_phrases += len(re.findall(pat, text))

    valid_used = [i for i in used if i in allowed_idx]
    require_cites = bool(contexts)
    ok = (
        not invalid_brackets
        and not fake_pmids
        and not fake_ncts
        and not fake_docs
        and (bool(valid_used) if require_cites else True)
    )
    return {
        "ok": ok,
        "used_brackets": used,
        "valid_used_brackets": valid_used,
        "invalid_brackets": invalid_brackets,
        "claimed_pmids": claimed_pmids,
        "fake_pmids": fake_pmids,
        "claimed_ncts": claimed_ncts,
        "fake_ncts": fake_ncts,
        "fake_docs": fake_docs,
        "has_citations": bool(used),
        "body_has_citations": bool(valid_used),
        "invent_phrase_hits": invent_phrases,
    }


def strip_invalid_claims(answer: str, check: dict[str, Any]) -> str:
    """
    校验失败时，在回答末尾追加「无法核实」提示（不直接删除全文）。

    参数:
        answer: 原回答。
        check: verify_citations 返回值。

    返回:
        str: 可能追加提示后的回答。
    """
    if check.get("ok", True):
        return answer
    notes = []
    if check.get("invalid_brackets"):
        notes.append(f"无效引用编号: {check['invalid_brackets']}")
    if not check.get("body_has_citations") and check.get("has_citations") is False:
        notes.append("正文未出现合法 [n] 引用")
    elif not check.get("body_has_citations"):
        notes.append("正文缺少可核对的 [n] 引用")
    if check.get("fake_pmids"):
        notes.append(f"无法核实的 PMID: {check['fake_pmids']}")
    if check.get("fake_ncts"):
        notes.append(f"无法核实的 NCT: {check['fake_ncts']}")
    if check.get("fake_docs"):
        notes.append(f"无法核实的文档 ID: {check['fake_docs']}")
    note = "；".join(notes) if notes else "检测到可疑引用"
    return (
        answer.rstrip()
        + f"\n\n> 引用校验：{note}。请仅采信与证据面板一致的引用。"
    )


# ---------------------------------------------------------------------------
# 【待完善】引用修复与幻觉防御增强（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def repair_answer_with_valid_cites(
    answer: str,
    contexts: list[dict[str, Any]],
    check: dict[str, Any],
) -> str:
    from src.tools.citation_policy import repair_answer_with_valid_cites as _repair

    return _repair(answer, contexts, check)


def detect_unsupported_claims(
    answer: str,
    contexts: list[dict[str, Any]],
) -> list[str]:
    from src.tools.citation_policy import detect_unsupported_claims as _detect

    return _detect(answer, contexts)
