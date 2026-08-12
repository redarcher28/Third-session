# -*- coding: utf-8 -*-
"""
Chroma 向量仓储：负责 chunk 入库、语义检索、导出 BM25 语料。
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings
from src.ingest import load_docs
from src.kb.chunking import docs_to_chunks, merge_tiny_chunks
from src.llm import get_llm
from src.models import Chunk

logger = logging.getLogger(__name__)

COLLECTION = "evidence_chunks"
COLLECTION_STAGING = "evidence_chunks_staging"
BM25_CACHE_NAME = "bm25_corpus.json"
# 检索侧 BM25 与统计导出应读取全量语料，避免「库有 11k、只检 5k」
BM25_RETRIEVAL_LIMIT = 100_000
BM25_EXPORT_LIMIT = 100_000


def _bm25_cache_path() -> Path:
    return get_settings().processed_path / BM25_CACHE_NAME


def _load_bm25_cache(limit: int = BM25_RETRIEVAL_LIMIT) -> list[dict[str, Any]]:
    path = _bm25_cache_path()
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("BM25 cache read failed: %s", exc)
        return []
    if not isinstance(rows, list):
        return []
    return rows[:limit]


def export_bm25_cache(docs: list[dict[str, Any]]) -> Path:
    """将 BM25 语料写入 sidecar，Chroma 不可用时仍可检索。"""
    path = _bm25_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    logger.info("BM25 cache exported: %d rows -> %s", len(docs), path)
    return path


class EvidenceStore:
    """证据向量库封装。"""

    def __init__(self, *, collection_name: str = COLLECTION) -> None:
        """打开（或创建）持久化 Chroma 集合。"""
        settings = get_settings()
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """删除并重建集合（建库脚本默认会调用）。"""
        name = self._collection_name
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def chroma_count(self) -> tuple[int, bool]:
        """返回 (条数, 是否来自 Chroma 真读)。False 表示读取失败。"""
        try:
            return self._col.count(), True
        except Exception:
            return 0, False

    def retrieval_count(self) -> int:
        """BM25 侧车/cache 可见条数（Chroma 不可用时的检索规模）。"""
        try:
            return len(self.all_chunks_for_bm25(limit=BM25_RETRIEVAL_LIMIT))
        except Exception:
            return len(_load_bm25_cache(limit=BM25_RETRIEVAL_LIMIT))

    def count_detail(self) -> dict[str, Any]:
        """区分 Chroma 真 count 与 BM25 回退 count。"""
        n, from_chroma = self.chroma_count()
        if from_chroma:
            return {"count": n, "source": "chroma", "chroma_ok": True}
        cached = _load_bm25_cache(limit=BM25_EXPORT_LIMIT)
        return {
            "count": len(cached),
            "source": "bm25_cache",
            "chroma_ok": False,
        }

    def count(self) -> int:
        """返回当前集合中的向量条数；Chroma 异常时回退 BM25 缓存条数。"""
        detail = self.count_detail()
        if detail["source"] == "bm25_cache" and detail["count"]:
            logger.warning(
                "Chroma count failed; using BM25 cache (%d)",
                detail["count"],
            )
        if detail["count"] == 0 and not detail["chroma_ok"]:
            raise RuntimeError("Chroma count failed and BM25 cache is empty")
        return int(detail["count"])

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
                extra = c.extra or {}
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
                        "record_type": c.record_type,
                        "citation_eligible": bool(c.citation_eligible),
                        "source_locator": (c.source_locator or c.doc_id)[:500],
                        "trial_status": str(extra.get("status") or "")[:120],
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
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        语义检索 Top-N。

        参数:
            query: 查询文本。
            n_results: 返回条数。
            tag_filter: 可选标签过滤（部分 Chroma 版本可能不支持，失败则回退）。
            source: 可选来源过滤（如 "wiki"），比 $contains 更可靠。

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
        where: dict[str, Any] = {}
        if tag_filter:
            where["tags"] = {"$contains": tag_filter}
        if source:
            where["source"] = source
        if where:
            kwargs["where"] = where
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

    def all_chunks_for_bm25(self, limit: int = BM25_RETRIEVAL_LIMIT) -> list[dict[str, Any]]:
        """
        导出集合中的文本块，供 BM25 关键词检索使用。

        参数:
            limit: 最多导出条数。

        返回:
            list[dict]: 含 chunk_id/text 及元数据。
        """
        try:
            count = self._col.count()
            if count == 0:
                return _load_bm25_cache(limit=limit)
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
        except Exception as exc:
            cached = _load_bm25_cache(limit=limit)
            if cached:
                logger.warning(
                    "Chroma export failed (%s); using BM25 cache (%d rows)",
                    type(exc).__name__,
                    len(cached),
                )
                return cached
            raise


class BuildValidationError(ValueError):
    """建库候选数据未通过硬校验，禁止覆盖正式库。"""


def _write_index_manifest(chunk_count: int) -> None:
    import importlib.metadata

    manifest = {
        "chunk_count": chunk_count,
        "collection": COLLECTION,
        "chromadb_version": importlib.metadata.version("chromadb"),
    }
    path = get_settings().processed_path / "index_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Index manifest -> %s", path)


def validate_build_chunks(
    chunks: list[Chunk],
    *,
    previous_count: int | None = None,
    min_absolute: int = 100,
    min_ratio: float = 0.5,
) -> dict[str, Any]:
    """建库前硬校验：溯源失败或数量异常骤降则拒绝发布。"""
    from src.kb.chunking import validate_chunk_traceability

    trace = validate_chunk_traceability(chunks)
    if not trace["ok"]:
        raise BuildValidationError(f"chunk traceability failed: {trace}")
    if len(chunks) < min_absolute:
        raise BuildValidationError(f"too few chunks to publish: {len(chunks)} < {min_absolute}")
    if previous_count and previous_count >= min_absolute:
        if len(chunks) < int(previous_count * min_ratio):
            raise BuildValidationError(
                f"chunk count dropped suspiciously: {len(chunks)} vs previous {previous_count}"
            )
    return trace


def _copy_collection_with_embeddings(
    source: EvidenceStore,
    target: EvidenceStore,
    *,
    batch_size: int = 128,
) -> int:
    """将 source 集合（含 embedding）复制到 target。"""
    res = source._col.get(include=["embeddings", "documents", "metadatas"])
    ids = res.get("ids") or []
    if not ids:
        return 0
    total = 0
    for i in range(0, len(ids), batch_size):
        sl = slice(i, i + batch_size)
        target._col.upsert(
            ids=ids[sl],
            embeddings=res["embeddings"][sl],
            documents=res["documents"][sl],
            metadatas=res["metadatas"][sl],
        )
        total += len(ids[sl])
    return total


def atomic_publish_chunks(
    chunks: list[Chunk],
    *,
    previous_count: int | None = None,
) -> int:
    """
    在 staging 集合完成写入与探针，通过后原子切换为正式库；失败则保留旧库。
    """
    from src.kb.health import probe_chroma

    validate_build_chunks(chunks, previous_count=previous_count)
    staging = EvidenceStore(collection_name=COLLECTION_STAGING)
    staging.reset()
    n = staging.upsert_chunks(chunks)
    staging_probe = probe_chroma(staging)
    if not staging_probe["chroma_ok"] or staging_probe["store_count"] != n:
        staging.reset()
        raise BuildValidationError(f"staging Chroma probe failed: {staging_probe}")

    main = EvidenceStore()
    new_ids = {c.chunk_id for c in chunks}
    try:
        copied = _copy_collection_with_embeddings(staging, main)
        if copied != n:
            raise BuildValidationError(f"staging->main copy mismatch: copied={copied} expected={n}")

        main_probe = probe_chroma(main)
        if not main_probe["chroma_ok"] or main_probe["store_count"] < n:
            raise BuildValidationError(f"main collection probe failed after publish: {main_probe}")

        try:
            existing_ids = main._col.get(include=[]).get("ids") or []
            stale = [cid for cid in existing_ids if cid not in new_ids]
            if stale:
                main._col.delete(ids=stale)
                logger.info("Pruned %d stale chunks after atomic publish", len(stale))
        except Exception as exc:
            logger.warning("Stale prune after publish skipped: %s", exc)
    except BuildValidationError:
        try:
            if new_ids:
                main._col.delete(ids=list(new_ids))
        except Exception as exc:
            logger.warning("Rollback of partial publish failed: %s", exc)
        raise
    finally:
        staging.reset()
    try:
        export_bm25_cache(main.all_chunks_for_bm25(limit=BM25_EXPORT_LIMIT))
    except Exception as exc:
        logger.warning("BM25 cache export skipped after publish: %s", exc)
    _write_index_manifest(n)
    logger.info("Atomic publish complete: %d chunks (main count=%d)", n, main.count())
    return n


# ---------------------------------------------------------------------------
# 向量库运维与重建
# ---------------------------------------------------------------------------


def _existing_items(store: EvidenceStore) -> dict[str, dict]:
    """导出集合中现有 chunk_id -> {text, doc_id} 映射，供增量重建比对与剪枝。"""
    items = store.all_chunks_for_bm25(limit=BM25_EXPORT_LIMIT)
    return {
        m["chunk_id"]: {"text": m.get("text", ""), "doc_id": m.get("doc_id", "")}
        for m in items
    }


def rebuild_collection_from_processed(reset: bool = True) -> int:
    """
    仅从 data/processed 已有文件重建向量索引，不重新联网采集。

    创新点：
        - 增量模式（reset=False）：以「文档文本指纹」比对，正文未变的 chunk 直接跳过，
          只重算变化/新增内容，改切分参数或加料后秒级更新；
        - 删除剪枝：processed 中已不存在的文档，其 chunk 自动从库里清除，
          增量更新真正做到「增改删」闭环，而非只增不删；
        - 自动选择 documents_with_wiki.json，缺失时回退 documents.json。

    参数:
        reset: 是否先清空再写入；默认 True。

    返回:
        int: 成功入库的 chunk 数量。

    作用:
        改切分策略或 embedding 后快速重建，缩短联调时间。
    """
    settings = get_settings()
    merged = settings.processed_path / "documents_with_wiki.json"
    if not merged.exists():
        merged = settings.processed_path / "documents.json"
    docs = load_docs(merged)
    # 与 build_kb 流水线保持一致（同样经过短块合并），保证 chunk_id 口径相同
    chunks = merge_tiny_chunks(docs_to_chunks(docs))
    store = EvidenceStore()
    if reset:
        try:
            previous_count = store.count()
        except Exception:
            previous_count = None
        return atomic_publish_chunks(chunks, previous_count=previous_count)

    full_chunks = chunks  # 全量快照：用于判断「文档是否还存在于 processed」
    validate_build_chunks(full_chunks, previous_count=store.count())
    existing = _existing_items(store)
    before = len(full_chunks)
    chunks = [c for c in full_chunks if existing.get(c.chunk_id, {}).get("text") != c.text]
    if chunks:
        logger.info(
            "Incremental rebuild: %d changed/new chunks (skipped %d unchanged)",
            len(chunks),
            before - len(chunks),
        )
    live_doc_ids = {c.doc_id for c in full_chunks}
    stale = [cid for cid, m in existing.items() if m.get("doc_id") not in live_doc_ids]
    if stale:
        snapshot_count = store.count()
        if snapshot_count >= 100 and len(stale) > int(snapshot_count * 0.5):
            raise BuildValidationError(
                f"incremental prune blocked: would delete {len(stale)}/{snapshot_count} chunks"
            )
        store._col.delete(ids=stale)
        logger.info("Pruned %d stale chunks (docs removed from processed)", len(stale))
    if not chunks:
        logger.info("Incremental rebuild: nothing changed, store untouched (count=%d)", store.count())
        return 0
    n = store.upsert_chunks(chunks)
    try:
        export_bm25_cache(store.all_chunks_for_bm25(limit=BM25_EXPORT_LIMIT))
    except Exception as exc:
        logger.warning("BM25 cache export skipped: %s", exc)
    logger.info("Rebuilt knowledge base: %d chunks upserted (store count=%d)", n, store.count())
    return n


def export_store_stats(out_path: Path | None = None) -> dict:
    """
    导出当前向量库统计信息（总量、来源分布、证据等级分布、年份分布）。

    创新点：
        - 覆盖文档数 docs_covered：区分「chunk 数」与「去重后文献数」，
          演示时更能体现知识库真实规模；
        - 可选双格式输出：JSON（机器可读）+ Markdown（报告直接贴）。

    参数:
        out_path: 可选 Path；若提供则同时写入 .json 与 .md。

    返回:
        dict: 统计结果，含 count / docs_covered / by_source / by_level / by_year。

    作用:
        演示与报告中展示知识库规模与构成。
    """
    store = EvidenceStore()
    items = store.all_chunks_for_bm25(limit=BM25_EXPORT_LIMIT)
    by_source = Counter(m.get("source", "?") for m in items)
    by_level = Counter(m.get("evidence_level", "other") for m in items)
    by_year = Counter(m.get("year", "unknown") for m in items)
    docs_covered = {m.get("doc_id") for m in items}
    stats: dict[str, Any] = {
        "count": len(items),
        "docs_covered": len(docs_covered),
        "by_source": dict(by_source),
        "by_level": dict(by_level),
        "by_year": dict(sorted(by_year.items(), key=lambda kv: (kv[0] == "unknown", str(kv[0])))),
    }
    try:
        from src.kb.health import kb_health_report

        health = kb_health_report()
        stats["kb_status"] = health["status"]
        stats["chroma_ok"] = health["chroma"]["chroma_ok"]
        stats["degraded_reasons"] = health["degraded_reasons"]
        stats["retrieval_indexed_count"] = health["retrieval"]["bm25_indexed_count"]
        stats["bm25_cache_total"] = health["retrieval"]["bm25_cache_total"]
        stats["bm25_index_complete"] = health["retrieval"]["bm25_index_complete"]
    except Exception:
        pass
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path = out_path.with_suffix(".md")
        lines = ["# 向量库统计", "", f"- chunk 总数: {stats['count']}", f"- 覆盖文档数: {stats['docs_covered']}"]
        lines += ["", "## 来源分布", ""] + [f"- {k}: {v}" for k, v in sorted(by_source.items())]
        lines += ["", "## 证据等级分布", ""] + [f"- {k}: {v}" for k, v in sorted(by_level.items())]
        lines += ["", "## 年份分布", ""] + [f"- {k}: {v}" for k, v in stats["by_year"].items()]
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Store stats -> %s / %s", out_path, md_path)
    return stats
