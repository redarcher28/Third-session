# -*- coding: utf-8 -*-
"""
Chroma 向量仓储：负责 chunk 入库、语义检索、导出 BM25 语料。
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings
from src.llm import get_llm
from src.models import Chunk

logger = logging.getLogger(__name__)

COLLECTION = "evidence_chunks"


class EvidenceStore:
    """证据向量库封装。"""

    def __init__(self) -> None:
        """打开（或创建）持久化 Chroma 集合。"""
        settings = get_settings()
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """删除并重建集合（建库脚本默认会调用）。"""
        try:
            self._client.delete_collection(COLLECTION)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """返回当前集合中的向量条数。"""
        return self._col.count()

    def upsert_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> int:
        """
        批量向量化并写入/更新 chunks。

        参数:
            chunks: 切块列表。
            batch_size: 每批 embedding 条数。

        返回:
            int: 成功处理的块数。
        """
        if not chunks:
            return 0
        llm = get_llm()
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = llm.embed(texts)
            ids = [c.chunk_id for c in batch]
            metadatas: list[dict[str, Any]] = []
            documents: list[str] = []
            for c in batch:
                metadatas.append(
                    {
                        "doc_id": c.doc_id,
                        "source": c.source,
                        "title": c.title[:500],
                        "year": c.year if c.year is not None else -1,
                        "url": c.url or "",
                        "tags": ",".join(c.tags),
                        "evidence_level": c.evidence_level,
                        "chunk_index": c.chunk_index,
                    }
                )
                documents.append(c.text)
            self._col.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            total += len(batch)
            logger.info("Upserted %d/%d chunks", total, len(chunks))
        return total

    def query(
        self,
        query: str,
        *,
        n_results: int = 10,
        tag_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        语义检索 Top-N。

        参数:
            query: 查询文本。
            n_results: 返回条数。
            tag_filter: 可选标签过滤（部分 Chroma 版本可能不支持，失败则回退）。

        返回:
            list[dict]: 每项含 chunk_id/text/distance 及元数据字段。
        """
        llm = get_llm()
        emb = llm.embed([query])[0]
        kwargs: dict[str, Any] = {
            "query_embeddings": [emb],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if tag_filter:
            kwargs["where"] = {"tags": {"$contains": tag_filter}}
        try:
            res = self._col.query(**kwargs)
        except Exception:
            # some chroma versions dislike $contains on string; fall back
            res = self._col.query(
                query_embeddings=[emb],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        out: list[dict[str, Any]] = []
        if not res or not res.get("ids"):
            return out
        for idx, chunk_id in enumerate(res["ids"][0]):
            meta = (res["metadatas"][0][idx] or {}) if res.get("metadatas") else {}
            doc = (res["documents"][0][idx] or "") if res.get("documents") else ""
            dist = (res["distances"][0][idx] if res.get("distances") else None)
            out.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "distance": dist,
                    **meta,
                }
            )
        return out

    def all_chunks_for_bm25(self, limit: int = 5000) -> list[dict[str, Any]]:
        """
        导出集合中的文本块，供 BM25 关键词检索使用。

        参数:
            limit: 最多导出条数。

        返回:
            list[dict]: 含 chunk_id/text 及元数据。
        """
        count = self._col.count()
        if count == 0:
            return []
        n = min(count, limit)
        res = self._col.get(limit=n, include=["documents", "metadatas"])
        out = []
        for i, cid in enumerate(res["ids"]):
            meta = res["metadatas"][i] if res["metadatas"] else {}
            out.append(
                {
                    "chunk_id": cid,
                    "text": res["documents"][i] if res["documents"] else "",
                    **(meta or {}),
                }
            )
        return out


# ---------------------------------------------------------------------------
# 【待完善】向量库运维与重建（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def rebuild_collection_from_processed(reset: bool = True) -> int:
    """
    【待完善】仅从 data/processed 已有文件重建向量索引，不重新联网采集。

    参数:
        reset: 是否先清空再写入；默认 True。

    返回:
        int: 成功入库的 chunk 数量。

    作用:
        改切分策略或 embedding 后快速重建，缩短联调时间。
    """
    raise NotImplementedError("待队员实现：rebuild_collection_from_processed")


def export_store_stats(out_path=None) -> dict:
    """
    【待完善】导出当前向量库统计信息（总量、来源分布、标签分布）。

    参数:
        out_path: 可选 Path；若提供则同时写入 JSON/Markdown。

    返回:
        dict: 统计结果，建议含 count / by_source / by_level。

    作用:
        演示与报告中展示知识库规模与构成。
    """
    raise NotImplementedError("待队员实现：export_store_stats")
