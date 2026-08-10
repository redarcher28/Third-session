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
    raise NotImplementedError("待队员实现：validate_chunk_traceability")


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
    raise NotImplementedError("待队员实现：merge_tiny_chunks")
