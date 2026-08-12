# -*- coding: utf-8 -*-
"""知识库统计接口（api.py / web_server.py 共用）。"""

from __future__ import annotations

from src.kb.store import export_store_stats


def fetch_kb_stats() -> dict:
    """导出并返回当前向量库统计；知识库未构建时不阻塞健康检查。"""
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
        }
    return {"ok": True, **stats}
