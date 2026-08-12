# -*- coding: utf-8 -*-
"""引用校验失败时的修复与降级策略。"""

from __future__ import annotations

import re
from typing import Any

from src.generation.answer import (
    DISCLAIMER,
    _evidence_only_fallback,
    contexts_to_citations,
    extract_citation_indices,
)
from src.models import Citation
from src.text_utils import filter_citable_contexts
from src.tools.cite_check import answer_body_for_citation_check, verify_citations


def repair_answer_with_valid_cites(
    answer: str,
    contexts: list[dict[str, Any]],
    check: dict[str, Any],
) -> str:
    """校验失败时，移除不可信正文并退回 evidence-only 列表。"""
    _ = answer, check
    citable = filter_citable_contexts(contexts)
    if not citable:
        return ""
    return _evidence_only_fallback(contexts_to_citations(citable))


def detect_unsupported_claims(
    answer: str,
    contexts: list[dict[str, Any]],
) -> list[str]:
    """找出正文中缺少 [n] 且与证据词重叠很低的长句。"""
    body = answer_body_for_citation_check(answer)
    if not body.strip():
        return []
    cited = set(extract_citation_indices(body))
    blob = " ".join(
        str(c.get("text") or c.get("title") or "")[:500] for c in contexts[:5]
    ).lower()
    suspicious: list[str] = []
    for sent in re.split(r"(?<=[。！？.!?])\s*", body):
        s = sent.strip()
        if len(s) < 24:
            continue
        if cited and any(f"[{n}]" in s for n in cited):
            continue
        tokens = set(re.findall(r"[\w\u4e00-\u9fff]{2,}", s.lower()))
        evidence_tokens = set(re.findall(r"[\w\u4e00-\u9fff]{2,}", blob))
        overlap = len(tokens & evidence_tokens) / max(1, len(tokens))
        if overlap < 0.08:
            suspicious.append(s[:240])
    return suspicious[:5]


def apply_citation_failure_policy(
    answer: str,
    contexts: list[dict[str, Any]],
    citations: list[Citation],
    check: dict[str, Any],
    *,
    refused: bool = False,
) -> tuple[str, list[Citation], dict[str, Any]]:
    """
    引用校验失败时的统一降级：
    1) 尝试 evidence-only 重写；2) 仍失败则保留拒答式说明。
    """
    if refused or check.get("ok"):
        return answer, citations, check

    repaired = repair_answer_with_valid_cites(answer, contexts, check)
    if repaired:
        citable = filter_citable_contexts(contexts)
        new_check = verify_citations(repaired, citable)
        cites = contexts_to_citations(citable)
        if new_check.get("ok"):
            if DISCLAIMER.strip() not in repaired:
                repaired = repaired.rstrip() + DISCLAIMER
            return repaired, cites, new_check

    unsupported = detect_unsupported_claims(answer, contexts)
    reason = check.get("reason") or "citation_check_failed"
    failed_check = {
        **check,
        "ok": False,
        "reason": reason,
        "unsupported_claims": unsupported,
    }
    fallback = repair_answer_with_valid_cites(answer, contexts, check)
    if fallback:
        if DISCLAIMER.strip() not in fallback:
            fallback = fallback.rstrip() + DISCLAIMER
        return fallback, [], failed_check
    return (
        "本次回答未能通过引用校验，系统已阻止展示未经验证的综合结论。请查看证据面板中的检索片段。"
        + DISCLAIMER,
        [],
        failed_check,
    )
