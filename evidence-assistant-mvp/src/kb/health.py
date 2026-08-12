# -*- coding: utf-8 -*-
"""知识库/Chroma 健康探测：区分「库里有数据」与「检索链路真实可用」。"""

from __future__ import annotations

import importlib.metadata
import json
import logging
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.kb.store import (
    BM25_EXPORT_LIMIT,
    BM25_RETRIEVAL_LIMIT,
    COLLECTION,
    EvidenceStore,
    _load_bm25_cache,
)

logger = logging.getLogger(__name__)


def _chromadb_version() -> str:
    try:
        return importlib.metadata.version("chromadb")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def load_index_manifest() -> dict[str, Any] | None:
    path = get_settings().processed_path / "index_manifest.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("Index manifest read failed: %s", exc)
        return None


def validate_index_manifest(chroma_store_count: int) -> dict[str, Any]:
    """比对 manifest 与当前运行时版本/条数。"""
    manifest = load_index_manifest()
    if not manifest:
        return {"ok": False, "reason": "manifest_missing", "manifest": None}
    expected_version = str(manifest.get("chromadb_version") or "")
    runtime_version = _chromadb_version()
    expected_count = int(manifest.get("chunk_count") or 0)
    version_ok = not expected_version or expected_version == runtime_version
    count_ok = expected_count <= 0 or abs(chroma_store_count - expected_count) <= max(5, int(expected_count * 0.01))
    ok = version_ok and count_ok
    reasons: list[str] = []
    if not version_ok:
        reasons.append("manifest_version_mismatch")
    if not count_ok:
        reasons.append("manifest_count_mismatch")
    return {
        "ok": ok,
        "reasons": reasons,
        "manifest": manifest,
        "runtime_chromadb_version": runtime_version,
        "runtime_store_count": chroma_store_count,
    }


def probe_chroma(store: EvidenceStore | None = None) -> dict[str, Any]:
    """
    探测 Chroma count / get / query 是否可用。

    返回字段:
        chroma_ok, count_ok, get_ok, query_ok, store_count, errors
    """
    store = store or EvidenceStore()
    errors: list[str] = []
    count_ok = get_ok = query_ok = False
    store_count = 0

    try:
        store_count = store._col.count()
        count_ok = True
    except Exception as exc:
        errors.append(f"count:{type(exc).__name__}:{exc}")

    if count_ok and store_count > 0:
        try:
            res = store._col.get(limit=1, include=["documents", "metadatas"])
            get_ok = bool(res and res.get("ids"))
        except Exception as exc:
            errors.append(f"get:{type(exc).__name__}:{exc}")

        try:
            from src.llm import get_llm

            llm = get_llm()
            emb = llm.embed(["hypertension guideline blood pressure"])[0]
            store._col.query(
                query_embeddings=[emb],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )
            query_ok = True
        except Exception as exc:
            errors.append(f"query:{type(exc).__name__}:{exc}")
    elif count_ok and store_count == 0:
        get_ok = query_ok = True

    chroma_ok = count_ok and get_ok and query_ok
    return {
        "chroma_ok": chroma_ok,
        "count_ok": count_ok,
        "get_ok": get_ok,
        "query_ok": query_ok,
        "store_count": store_count,
        "collection": store._collection_name,
        "chromadb_version": _chromadb_version(),
        "errors": errors,
    }


def probe_retrieval_index() -> dict[str, Any]:
    """BM25 检索索引实际加载条数 vs 缓存/库总量。"""
    cache_rows = _load_bm25_cache(limit=BM25_EXPORT_LIMIT)
    cache_total = len(cache_rows)
    try:
        indexed = len(EvidenceStore().all_chunks_for_bm25(limit=BM25_RETRIEVAL_LIMIT))
    except Exception as exc:
        logger.warning("Retrieval index probe failed: %s", exc)
        indexed = len(_load_bm25_cache(limit=BM25_RETRIEVAL_LIMIT))
    return {
        "bm25_cache_total": cache_total,
        "bm25_indexed_count": indexed,
        "bm25_index_complete": cache_total == 0 or indexed >= cache_total,
    }


def kb_health_report() -> dict[str, Any]:
    """汇总 /health 与 /kb/stats 共用的健康报告。"""
    chroma = probe_chroma()
    retrieval = probe_retrieval_index()
    manifest = validate_index_manifest(chroma["store_count"])
    degraded_reasons: list[str] = []
    if not chroma["chroma_ok"]:
        degraded_reasons.append("chroma_unavailable")
    if chroma["store_count"] == 0 and retrieval["bm25_cache_total"] == 0:
        degraded_reasons.append("empty_knowledge_base")
    if not retrieval["bm25_index_complete"]:
        degraded_reasons.append("bm25_partial_index")
    if manifest and not manifest["ok"]:
        reasons = list(manifest.get("reasons") or [])
        if not reasons and manifest.get("reason"):
            reasons = [str(manifest["reason"])]
        degraded_reasons.extend(reasons or ["manifest_mismatch"])

    status = "ok" if not degraded_reasons else "degraded"
    return {
        "status": status,
        "degraded_reasons": degraded_reasons,
        "chroma": chroma,
        "retrieval": retrieval,
        "manifest": manifest,
    }
