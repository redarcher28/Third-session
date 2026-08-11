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
# 切分质量与溯源校验
# ---------------------------------------------------------------------------


def validate_chunk_traceability(chunks: list[Chunk]) -> dict:
    """
    检查每个切块是否具备可追溯字段。

    创新点：
        - 溯源三元组（doc_id / title / url）全量校验：缺一项即报告具体 chunk_id；
        - 跨字段一致性：chunk_id 必须以 doc_id 开头（「doc_id#cN」规范），
          防止引用时对不上原文；
        - 追加空正文检查与总量统计，便于在 build 流程里做门槛断言。

    参数:
        chunks: 切分后的块列表。

    返回:
        dict: 例如 {
            "ok": bool,
            "total": int,
            "missing_doc_id": int,
            "missing_title": int,
            "missing_url": int,
            "missing_text": int,
            "bad_ids": list[str],
            "mismatched_ids": list[str],
        }

    作用:
        防止「有回答但点不开/对不上文献」的演示事故。
    """
    missing_doc_id = [c.chunk_id for c in chunks if not (c.doc_id or "").strip()]
    missing_title = [c.chunk_id for c in chunks if not (c.title or "").strip()]
    missing_url = [c.chunk_id for c in chunks if not (c.url or "").strip()]
    missing_text = [c.chunk_id for c in chunks if not (c.text or "").strip()]
    bad_ids = [
        c.chunk_id for c in chunks
        if not (c.chunk_id or "").strip() or "#" not in c.chunk_id
    ]
    mismatched_ids = [
        c.chunk_id for c in chunks
        if c.doc_id and c.chunk_id and not c.chunk_id.startswith(c.doc_id)
    ]
    report = {
        "ok": not (missing_doc_id or missing_title or missing_text or bad_ids or mismatched_ids),
        "total": len(chunks),
        "missing_doc_id": len(missing_doc_id),
        "missing_title": len(missing_title),
        "missing_url": len(missing_url),
        "missing_text": len(missing_text),
        "bad_ids": bad_ids[:10],
        "mismatched_ids": mismatched_ids[:10],
        "sample_missing_url": missing_url[:5],
    }
    return report


def merge_tiny_chunks(chunks: list[Chunk], min_chars: int = 120) -> list[Chunk]:
    """
    合并过短切块，减少碎片化检索命中。

    创新点：
        - 与「相邻且同源（同 doc_id）」块合并，绝不跨文档拼接，保证溯源不串；
        - 合并后统一重写 chunk_id / chunk_index，保持「doc_id#cN」溯源规范；
        - 贪心向后合并：当前块过短或上一块过短且同源时即合并，一次扫描完成。

    参数:
        chunks: 原始切块列表。
        min_chars: 低于该长度的块尝试与相邻块合并。

    返回:
        list[Chunk]: 合并后的切块列表（已重写 chunk_id / chunk_index）。

    作用:
        提升检索片段可读性，降低无意义短句干扰。
    """
    if not chunks:
        return []
    merged: list[Chunk] = []
    for c in chunks:
        if (
            merged
            and merged[-1].doc_id == c.doc_id
            and (len(merged[-1].text) < min_chars or len(c.text) < min_chars)
        ):
            merged[-1].text = f"{merged[-1].text} {c.text}".strip()
            # 标题/年份等以更完整者为准
            if not merged[-1].url and c.url:
                merged[-1].url = c.url
            if not merged[-1].year and c.year:
                merged[-1].year = c.year
        else:
            merged.append(c)
    # 按「每文档独立序号」重编号，保证 chunk_id 与模型语义一致且跨构建稳定
    counters: dict[str, int] = {}
    for c in merged:
        i = counters.get(c.doc_id, 0)
        counters[c.doc_id] = i + 1
        c.chunk_index = i
        c.chunk_id = f"{c.doc_id}#c{i}"
    return merged
