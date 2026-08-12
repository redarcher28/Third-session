# -*- coding: utf-8 -*-
"""知识库统计接口（api.py / web_server.py 共用）。"""

from __future__ import annotations

from src.kb.health import kb_health_report
from src.kb.store import export_store_stats


def fetch_kb_stats() -> dict:
    """导出并返回当前向量库统计；附带检索索引真实覆盖情况。"""
    health = kb_health_report()
    try:
        stats = export_store_stats()
    except Exception as exc:
        return {
            "ok": False,
            "count": 0,
            "docs_covered": 0,
            "by_source": {},
            "by_level": {},
            "error": str(exc),
            "kb_status": health["status"],
            "degraded_reasons": health["degraded_reasons"],
        }
    return {
        "ok": health["status"] == "ok",
        "kb_status": health["status"],
        "degraded_reasons": health["degraded_reasons"],
        "chroma_ok": health["chroma"]["chroma_ok"],
        "retrieval_indexed_count": health["retrieval"]["bm25_indexed_count"],
        "bm25_cache_total": health["retrieval"]["bm25_cache_total"],
        "bm25_index_complete": health["retrieval"]["bm25_index_complete"],
        **stats,
    }
