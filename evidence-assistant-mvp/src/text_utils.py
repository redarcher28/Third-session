# -*- coding: utf-8 -*-
"""文本展示工具：按句子/词边界截断，避免证据面板出现半句。"""

from __future__ import annotations

from typing import Any

# 中英文常见句读符（长者优先匹配由 rfind 自然处理）
_SENTENCE_SEPS = (
    "。",
    "！",
    "？",
    ". ",
    "! ",
    "? ",
    "；",
    "; ",
    "\n",
)

# 句读符找不到时的次级断点
_SOFT_SEPS = (" ", "，", ",", "、", "：", ":", "）", ")", "】", "]")


def truncate_at_sentence(
    text: str,
    max_chars: int,
    *,
    min_keep: int = 0,
    ellipsis: str = "…",
) -> str:
    """
    在 max_chars 以内尽量在完整句子末尾截断；必要时退到词/标点边界。

    参数:
        text: 原文。
        max_chars: 最大字符数（含省略号）。
        min_keep: 句子断点至少保留的前缀长度，避免截得过短。
        ellipsis: 确实被截断时追加的省略号。
    """
    cleaned = (text or "").strip()
    if not cleaned or len(cleaned) <= max_chars:
        return cleaned

    budget = max_chars - len(ellipsis) if ellipsis else max_chars
    if budget < 1:
        return ellipsis or cleaned[:max_chars]

    window = cleaned[:budget]
    best_end = -1
    for sep in _SENTENCE_SEPS:
        idx = window.rfind(sep)
        if idx >= min_keep and idx + len(sep) > best_end:
            best_end = idx + len(sep)

    if best_end > min_keep:
        return cleaned[:best_end].strip()

    for sep in _SOFT_SEPS:
        idx = window.rfind(sep)
        if idx >= min_keep:
            return cleaned[: idx + len(sep)].strip() + ellipsis

    return window.rstrip() + ellipsis


def citation_display_fields(
    raw_text: str,
    *,
    snippet_max: int = 320,
) -> tuple[str, str]:
    """
    生成证据展示用的 (完整文本, 短摘要)。

    完整文本供证据面板展开阅读；短摘要供参考文献等紧凑场景。
    """
    full = (raw_text or "").strip()
    if not full:
        return "", ""
    snippet = truncate_at_sentence(full, snippet_max, min_keep=max(40, snippet_max // 4))
    return full, snippet


def filter_citable_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """仅保留可作为正文引用的证据（citation_eligible=True）。"""
    return [c for c in contexts if c.get("citation_eligible", True)]


def context_to_citation_kwargs(context: dict[str, Any], index: int) -> dict[str, Any]:
    """从检索上下文 dict 构造 Citation 字段（含完整 text 与句子对齐 snippet）。"""
    raw_text = str(context.get("text") or "")
    full, snippet = citation_display_fields(raw_text)

    year = context.get("year")
    try:
        parsed_year = None if year in (None, -1, "-1") else int(year)
    except (TypeError, ValueError):
        parsed_year = None

    return {
        "index": index,
        "doc_id": str(context.get("doc_id") or context.get("chunk_id") or ""),
        "title": str(context.get("title") or ""),
        "source": str(context.get("source") or ""),
        "year": parsed_year,
        "url": str(context.get("url") or ""),
        "evidence_level": str(context.get("evidence_level") or "other"),
        "text": full,
        "snippet": snippet,
        "record_type": str(context.get("record_type") or "other"),
        "trial_status": str(context.get("trial_status") or ""),
        "citation_eligible": bool(context.get("citation_eligible", True)),
    }
