# -*- coding: utf-8 -*-
"""
文档切分模块：把 EvidenceDoc 切成可检索、可溯源的 Chunk。
"""

from __future__ import annotations

import re
from typing import Iterable

from src.models import Chunk, EvidenceDoc


def _split_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """
    按字符长度切分文本，尽量在句号处断开，并保留重叠窗口。

    参数:
        text: 原文。
        max_chars: 单块最大字符数。
        overlap: 相邻块重叠字符数。

    返回:
        list[str]: 文本块列表。
    """
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # prefer break at sentence
        if end < len(text):
            window = text[start:end]
            for sep in ["。", ". ", "；", "; ", "\n"]:
                idx = window.rfind(sep)
                if idx > max_chars // 3:
                    end = start + idx + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def docs_to_chunks(docs: Iterable[EvidenceDoc], max_chars: int = 1200) -> list[Chunk]:
    """
    将文档列表切分为 Chunk，并复制溯源元数据。

    参数:
        docs: EvidenceDoc 可迭代对象。
        max_chars: 单块最大字符数。

    返回:
        list[Chunk]: 切块结果；空正文时退化为标题块。
    """
    out: list[Chunk] = []
    for doc in docs:
        parts = _split_text(doc.text, max_chars=max_chars)
        if not parts:
            parts = [doc.title]
        for i, part in enumerate(parts):
            out.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#c{i}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    text=part,
                    year=doc.year,
                    url=doc.url,
                    tags=list(doc.tags),
                    evidence_level=doc.evidence_level,
                    chunk_index=i,
                )
            )
    return out


# ---------------------------------------------------------------------------
# 【待完善】切分质量与溯源校验（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def validate_chunk_traceability(chunks: list[Chunk]) -> dict:
    """
    【待完善】检查每个切块是否具备可追溯字段。

    参数:
        chunks: 切分后的块列表。

    返回:
        dict: 例如 {
            "ok": bool,
            "missing_doc_id": int,
            "missing_title": int,
            "missing_url": int,
            "bad_ids": list[str],
        }

    作用:
        防止「有回答但点不开/对不上文献」的演示事故。
    """
    missing_doc_id: list[str] = []
    missing_title: list[str] = []
    missing_text: list[str] = []
    missing_url: list[str] = []
    duplicate_ids: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        cid = chunk.chunk_id or "<empty>"
        if cid in seen:
            duplicate_ids.append(cid)
        seen.add(cid)
        if not chunk.doc_id.strip():
            missing_doc_id.append(cid)
        if not chunk.title.strip():
            missing_title.append(cid)
        if not chunk.text.strip():
            missing_text.append(cid)
        # local/wiki 样例允许没有外部 URL；其他公开源必须可追溯。
        if chunk.source not in ("local", "wiki") and not chunk.url.strip():
            missing_url.append(cid)

    return {
        "ok": not (missing_doc_id or missing_title or missing_text or missing_url or duplicate_ids),
        "total": len(chunks),
        "missing_doc_id": len(missing_doc_id),
        "missing_title": len(missing_title),
        "missing_text": len(missing_text),
        "missing_url": len(missing_url),
        "duplicate_ids": len(duplicate_ids),
        "bad_ids": sorted(set(missing_doc_id + missing_title + missing_text + missing_url + duplicate_ids))[:100],
    }


def merge_tiny_chunks(chunks: list[Chunk], min_chars: int = 120) -> list[Chunk]:
    """
    【待完善】合并过短切块，减少碎片化检索命中。

    参数:
        chunks: 原始切块列表。
        min_chars: 低于该长度的块尝试与相邻块合并。

    返回:
        list[Chunk]: 合并后的切块列表（需重写 chunk_id / chunk_index）。

    作用:
        提升检索片段可读性，降低无意义短句干扰。
    """
    if not chunks:
        return []

    by_doc: dict[str, list[Chunk]] = {}
    doc_order: list[str] = []
    for chunk in chunks:
        if chunk.doc_id not in by_doc:
            by_doc[chunk.doc_id] = []
            doc_order.append(chunk.doc_id)
        by_doc[chunk.doc_id].append(chunk)

    merged: list[Chunk] = []
    for doc_id in doc_order:
        group = sorted(by_doc[doc_id], key=lambda c: c.chunk_index)
        pending: Chunk | None = None
        doc_chunks: list[Chunk] = []
        for chunk in group:
            if pending is None:
                pending = chunk
                continue
            if len(pending.text.strip()) < min_chars:
                pending = pending.model_copy(
                    update={"text": f"{pending.text.strip()}\n\n{chunk.text.strip()}".strip()}
                )
            else:
                doc_chunks.append(pending)
                pending = chunk
        if pending is not None:
            if doc_chunks and len(pending.text.strip()) < min_chars:
                prev = doc_chunks.pop()
                pending = prev.model_copy(
                    update={"text": f"{prev.text.strip()}\n\n{pending.text.strip()}".strip()}
                )
            doc_chunks.append(pending)

        for new_index, chunk in enumerate(doc_chunks):
            merged.append(
                chunk.model_copy(
                    update={
                        "chunk_index": new_index,
                        "chunk_id": f"{chunk.doc_id}#c{new_index}",
                    }
                )
            )
    return merged
