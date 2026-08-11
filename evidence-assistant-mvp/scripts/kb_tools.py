# -*- coding: utf-8 -*-
"""
A 组知识库运维工具（B 组与演示直接使用）。

用法示例:
    python scripts/kb_tools.py stats                          # 导出统计 JSON+MD
    python scripts/kb_tools.py rebuild                        # 全量重建
    python scripts/kb_tools.py rebuild --incremental          # 增量重建（指纹跳过+剪枝）
    python scripts/kb_tools.py refresh-wiki                   # 列出可用主题
    python scripts/kb_tools.py refresh-wiki sodium-hypertension  # 刷新单个主题页
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/kb_tools.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kb.store import export_store_stats, rebuild_collection_from_processed  # noqa: E402
from src.kb.wiki import WIKI_TOPICS, refresh_single_wiki_page  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def cmd_stats(out: Path | None) -> None:
    """导出知识库统计并打印摘要。"""
    target = out or (Path("data") / "processed" / "store_stats")
    stats = export_store_stats(target)
    print(
        f"知识库统计: chunk={stats['count']} 去重后文档={stats['docs_covered']}\n"
        f"来源分布: {stats['by_source']}\n"
        f"等级分布: {stats['by_level']}"
    )


def cmd_rebuild(incremental: bool) -> None:
    """全量或增量重建向量库。"""
    n = rebuild_collection_from_processed(reset=not incremental)
    mode = "增量" if incremental else "全量"
    print(f"{mode}重建完成: 本轮入库 {n} 条")


def cmd_refresh(slug: str | None) -> None:
    """刷新单个 Wiki 主题页；不带 slug 时列出可选主题。"""
    if slug is None:
        print("可用 wiki 主题（slug: 标题）:")
        for t in WIKI_TOPICS:
            print(f"  - {t['slug']}: {t['title']}")
        return
    try:
        doc = refresh_single_wiki_page(slug)
        print(f"已刷新: {doc.doc_id}（{len(doc.text)} 字）")
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库运维与统计（A 组产出，B 组共用）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stats", help="导出知识库统计（JSON+MD 双格式）")
    p.add_argument("--out", type=Path, default=None, help="输出路径（不带扩展名）")

    p = sub.add_parser("rebuild", help="重建向量库")
    p.add_argument("--incremental", action="store_true", help="增量模式：指纹跳过未变 + 剪枝已删")

    p = sub.add_parser("refresh-wiki", help="刷新单个 Wiki 主题页")
    p.add_argument("slug", nargs="?", default=None, help="主题 slug；省略则列出可用主题")

    args = parser.parse_args()
    if args.cmd == "stats":
        cmd_stats(args.out)
    elif args.cmd == "rebuild":
        cmd_rebuild(args.incremental)
    elif args.cmd == "refresh-wiki":
        cmd_refresh(args.slug)


if __name__ == "__main__":
    main()
