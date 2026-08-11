# -*- coding: utf-8 -*-
"""
赛道二（健康营养助手）专项自检脚本。

对 data/eval/nutrition_questions.json 中的营养题逐题运行 ask()，
检查：科普大纲、引用存在性、引用校验、要点覆盖，并输出汇总。

用法:
    python scripts/check_nutrition.py [--questions data/eval/nutrition_questions.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tracks.pipeline import ask

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def _coverage(answer: str, gold_points: list[str]) -> float:
    """要点关键词覆盖率。"""
    if not gold_points:
        return 0.0
    hits = sum(1 for g in gold_points if g.lower() in answer.lower())
    return hits / len(gold_points)


def main() -> None:
    parser = argparse.ArgumentParser(description="赛道二营养自检")
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "eval" / "nutrition_questions.json",
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    args = parser.parse_args()

    items = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.limit > 0:
        items = items[: args.limit]

    rows: list[dict] = []
    violations = 0
    for it in items:
        resp = ask(it["question"], track="nutrition")
        check = resp.citation_check or {}
        outline = check.get("nutrition_outline") or {}
        cov = _coverage(resp.answer, it.get("gold_points") or [])
        has_cite = bool(check.get("has_citations"))
        cite_ok = bool(check.get("ok"))
        rows.append(
            {
                "id": it.get("id"),
                "refused": resp.refused,
                "n_ctx": len(resp.contexts),
                "cite": has_cite,
                "cite_ok": cite_ok,
                "outline": bool(outline),
                "coverage": round(cov, 2),
            }
        )
        must_cite = bool(it.get("must_cite", True))
        if must_cite and not resp.refused and (not has_cite or not cite_ok):
            violations += 1
        if not outline:
            violations += 1

    print(f"题目数: {len(rows)}")
    print("ID | 拒答 | 上下文 | 有引用 | 引用校验 | 大纲 | 要点覆盖")
    print("---|---|---|---|---|---|---")
    for r in rows:
        print(
            f"{r['id']} | {r['refused']} | {r['n_ctx']} | {r['cite']} | "
            f"{r['cite_ok']} | {r['outline']} | {r['coverage']}"
        )
    n_refused = sum(1 for r in rows if r["refused"])
    n_ok = sum(1 for r in rows if r["cite_ok"])
    print(
        f"汇总: 拒答 {n_refused}/{len(rows)} · 引用校验通过 {n_ok}/{len(rows)} · "
        f"问题数 {violations}"
    )
    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
