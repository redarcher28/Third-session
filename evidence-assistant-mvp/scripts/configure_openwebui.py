#!/usr/bin/env python3
"""把证据台的轻量产品信息合并进 Open WebUI 的持久化配置。

只写入 B 组自己标记的 banner / prompt suggestions，并保留用户已有的
Open WebUI 配置。脚本不读取、打印或修改任何 API token。
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


EVIDENCE_BANNER_ID = "evidence-assistant-mvp-boundary-v1"
EVIDENCE_WATERMARK = "证据智能助手 · 引用请人工复核 · 不构成医疗建议"

EVIDENCE_SUGGESTIONS: list[dict[str, Any]] = [
    {
        "title": ["赛道一 · 指南与 RCT", "高血压长期管理依据"],
        "content": "高血压患者为什么有时要长期吃药？有哪些指南或研究依据？",
    },
    {
        "title": ["赛道一 · 临床证据", "DASH 饮食证据"],
        "content": "DASH 饮食模式对血压的临床试验证据是什么？",
    },
    {
        "title": ["赛道二 · 研究提示", "地中海饮食"],
        "content": "地中海饮食对心血管风险有什么证据？",
    },
    {
        "title": ["赛道二 · 生活方式", "限钠饮食"],
        "content": "限钠饮食对高血压是否真的有帮助？",
    },
    {
        "title": ["赛道二 · 通俗解释", "血脂管理"],
        "content": "血脂高的人日常吃什么更有证据支持？",
    },
]


def evidence_banner(timestamp: int | None = None) -> dict[str, Any]:
    """构造旧证据台中最重要的双赛道说明。"""

    return {
        "id": EVIDENCE_BANNER_ID,
        "type": "info",
        "title": "证据台 · 先看证据，再形成答案",
        "content": (
            "**赛道一 · 临床证据**：指南、RCT、结局、适用人群 → 证据概览 + 关键来源；"
            "不替代临床决策，不给个体化处方或剂量。\n\n"
            "**赛道二 · 健康营养**：研究提示、生活方式、局限 → 通俗结论 + 边界 + 来源；"
            "不做个体化诊疗，不提供用药剂量。\n\n"
            "**RAG 链路**：查询改写 → Chroma + BM25 混合检索 → 基于证据生成 → 引用校验。"
        ),
        "dismissible": True,
        "timestamp": int(timestamp if timestamp is not None else time.time()),
    }


def _decode(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError):
        return fallback


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def merge_banners(existing: Any, *, timestamp: int | None = None) -> list[dict[str, Any]]:
    """把项目 banner 置顶，同时保留 Open WebUI/用户已有 banner。"""

    current = existing if isinstance(existing, list) else []
    rest = [
        item
        for item in current
        if not isinstance(item, dict) or item.get("id") != EVIDENCE_BANNER_ID
    ]
    return [evidence_banner(timestamp), *rest]


def merge_suggestions(existing: Any) -> list[dict[str, Any]]:
    """把旧证据台示例问题置顶，不删除 Open WebUI 的其他建议。"""

    current = existing if isinstance(existing, list) else []
    evidence_content = {item["content"] for item in EVIDENCE_SUGGESTIONS}
    rest = [
        item
        for item in current
        if not isinstance(item, dict) or item.get("content") not in evidence_content
    ]
    return [*EVIDENCE_SUGGESTIONS, *rest]


def _has_config_table(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'config'"
    ).fetchone()
    return row is not None


def apply_evidence_ui_config(
    db_path: Path,
    *,
    timestamp: int | None = None,
) -> dict[str, Any]:
    """合并 B 组 UI 配置，返回可审计但不含敏感信息的变更摘要。"""

    if not db_path.exists():
        return {"updated": False, "reason": "database_missing"}

    connection = sqlite3.connect(str(db_path), timeout=30)
    try:
        connection.execute("PRAGMA busy_timeout = 30000")
        if not _has_config_table(connection):
            return {"updated": False, "reason": "config_table_missing"}

        rows = connection.execute("SELECT key, value FROM config").fetchall()
        values = {key: _decode(value, None) for key, value in rows}
        banners = merge_banners(values.get("ui.banners"), timestamp=timestamp)
        suggestions = merge_suggestions(values.get("ui.prompt_suggestions"))

        updates: dict[str, Any] = {
            "ui.banners": banners,
            "ui.prompt_suggestions": suggestions,
            # 评测 Arena 不是本项目的模型，避免出现在双赛道模型选择器中。
            "evaluation.arena.enable": False,
        }
        if not values.get("ui.watermark"):
            updates["ui.watermark"] = EVIDENCE_WATERMARK

        now = int(timestamp if timestamp is not None else time.time())
        for key, value in updates.items():
            connection.execute(
                """
                INSERT INTO config (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, _encode(value), now),
            )
        connection.commit()
        return {
            "updated": True,
            "banner_count": len(banners),
            "suggestion_count": len(suggestions),
            "arena_disabled": True,
        }
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure the Evidence Desk Open WebUI layer")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Open WebUI DATA_DIR containing webui.db",
    )
    args = parser.parse_args()
    data_dir = args.data_dir or os.environ.get("OPENWEBUI_DATA_DIR")
    if not data_dir:
        parser.error("--data-dir or OPENWEBUI_DATA_DIR is required")

    result = apply_evidence_ui_config(Path(data_dir) / "webui.db")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
