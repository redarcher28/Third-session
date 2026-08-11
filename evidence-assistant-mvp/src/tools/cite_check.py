# -*- coding: utf-8 -*-
"""
引用校验工具：检测假引用、无效编号、编造 PMID/NCT。
"""

from __future__ import annotations

import re
from typing import Any

from src.generation.answer import extract_citation_indices


PMID_RE = re.compile(r"(?<![A-Za-z0-9])PMID[:\s]*([0-9]{5,9})(?![A-Za-z0-9])", re.I)
NCT_RE = re.compile(r"(?<![A-Za-z0-9])(NCT\d{8})(?![A-Za-z0-9])", re.I)
DOC_RE = re.compile(r"(?<![A-Za-z0-9])((?:pmid|nct|epmc|local|wiki):[^\s\]，。；;,]+)", re.I)


def verify_citations(answer: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    校验回答中的引用是否都能在当次证据中找到。

    参数:
        answer: 模型回答文本。
        contexts: 当次检索证据列表。

    返回:
        dict: 校验明细，关键字段包括：
            - ok (bool): 是否全部合法
            - used_brackets (list[int]): 使用的 [n]
            - invalid_brackets (list[int]): 越界编号
            - fake_pmids / fake_ncts / fake_docs: 无法核实的标识
            - has_citations (bool): 是否出现引用
    """
    allowed_idx = set(range(1, len(contexts) + 1))
    used = extract_citation_indices(answer)
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

    claimed_pmids = PMID_RE.findall(answer)
    claimed_ncts = [m.upper() for m in NCT_RE.findall(answer)]
    claimed_docs = [m.lower() for m in DOC_RE.findall(answer)]

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
        invent_phrases += len(re.findall(pat, answer))

    ok = not invalid_brackets and not fake_pmids and not fake_ncts and not fake_docs
    return {
        "ok": ok,
        "used_brackets": used,
        "invalid_brackets": invalid_brackets,
        "claimed_pmids": claimed_pmids,
        "fake_pmids": fake_pmids,
        "claimed_ncts": claimed_ncts,
        "fake_ncts": fake_ncts,
        "fake_docs": fake_docs,
        "has_citations": bool(used),
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
# 引用修复与幻觉防御增强
# ---------------------------------------------------------------------------


def repair_answer_with_valid_cites(
    answer: str,
    contexts: list[dict[str, Any]],
    check: dict[str, Any],
) -> str:
    """
    当引用校验失败时，基于合法证据重写回答中的引用部分。

    参数:
        answer: 原始生成回答。
        contexts: 当次检索证据。
        check: verify_citations 的返回结果。

    返回:
        str: 修复后的回答文本。

    作用:
        降低假引用残留，提升赛道三「假引用减少」指标表现。
    """
    allowed = set(range(1, len(contexts) + 1))

    def keep_valid(match: re.Match[str]) -> str:
        num = int(match.group(1))
        return match.group(0) if num in allowed else ""

    repaired = re.sub(r"\[(\d+)\]", keep_valid, answer)
    repaired = PMID_RE.sub("", repaired)
    repaired = NCT_RE.sub("", repaired)
    repaired = DOC_RE.sub("", repaired)
    # 清理移除编号后残留的 "PMID:" / "NCT" 等前缀
    repaired = re.sub(r"(?<![A-Za-z0-9])PMID[:\s]*", "", repaired, flags=re.I)
    repaired = re.sub(r"(?<![A-Za-z0-9])NCT\b", "", repaired, flags=re.I)
    repaired = re.sub(r"\s{2,}", " ", repaired).strip()

    if repaired != answer:
        repaired = (
            repaired.rstrip()
            + "\n\n> 引用校验：已移除无法核实的引用编号/文献标识，请以证据面板为准。"
        )
    return repaired


def detect_unsupported_claims(
    answer: str,
    contexts: list[dict[str, Any]],
) -> list[str]:
    """
    找出回答中可能未被证据支撑的句子/主张。

    参数:
        answer: 模型回答。
        contexts: 当次证据。

    返回:
        list[str]: 可疑主张列表（原文句子）。

    作用:
        辅助人工复核与自动拒答/降级策略。
    """
    allowed = set(range(1, len(contexts) + 1))
    sentences = re.split(r"(?<=[。！？!?；;])\s*|\n+", answer)
    suspicious: list[str] = []
    skip_markers = ("声明", "参考文献", "引用校验", "不构成", "仅供学习", "---")
    invent_patterns = [
        r"根据某项研究",
        r"著名的 .+ 试验",
        r"发表于《[^》]+》",
        r"et al\.",
    ]

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue
        if any(marker in sentence for marker in skip_markers):
            continue
        used = [int(x) for x in re.findall(r"\[(\d+)\]", sentence)]
        if any(i in allowed for i in used):
            continue
        if any(re.search(p, sentence) for p in invent_patterns):
            suspicious.append(sentence)
        elif re.search(r"[\u4e00-\u9fffA-Za-z]", sentence):
            suspicious.append(sentence)
        if len(suspicious) >= 10:
            break
    return suspicious
