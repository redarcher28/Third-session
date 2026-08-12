# -*- coding: utf-8 -*-
"""
固定题评估记录（D2AM 基础要求：≥8 题评估记录 + 四个固定）。

四固定：固定题集（data/eval/benchmark.jsonl）、固定配置（当前环境快照）、
固定评分口径（程序化检查 + 人工评审留空）、固定保留失败（不筛选记录）。

程序化检查（评四件事中的客观项）：
    - 引用与证据：must_cite 题引用是否非空、引用 id 是否真实存在于知识库（引用存在率）
    - 行为与边界：越界题（x1/x2）是否正确拒答
    - 回答质量：四维 Rubric（正确/完整/安全/清晰 1/3/5 分）留人工填写

输出:
    data/eval/results/assessment_records.jsonl（逐题记录）
    data/eval/results/assessment_summary.md（汇总表）

用法:
    python scripts/run_assessment.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Allow running as `python scripts/run_assessment.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings  # noqa: E402
from src.ingest import load_docs  # noqa: E402
from src.llm import get_llm  # noqa: E402
from src.tracks.pipeline import ask  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path("data/eval/benchmark.jsonl")
OUT_DIR = Path("data/eval/results")
# 越界/应拒答题（由 OUT_OF_SCOPE 关键词触发）
REFUSE_IDS = {"x1", "x2"}


def _config_snapshot() -> dict:
    """固定配置快照：记录模型/离线状态/知识库规模/日期。"""
    settings = get_settings()
    kb_docs = len(load_docs(settings.processed_path / "documents.json"))
    llm = get_llm()
    return {
        "offline": bool(llm.is_offline),
        "kb_docs": kb_docs,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "arm": "rag",
        "model": "offline-placeholder" if llm.is_offline else "configured",
    }


def _citation_exists(cite_doc_id: str, known_ids: set[str]) -> bool:
    """引用存在率检查：引用 id 必须真实存在于知识库（材料：ID/URL 拒答规则）。"""
    return cite_doc_id in known_ids


def _run_one(rec: dict, known_ids: set[str], cfg: dict) -> dict:
    qid = rec["id"]
    resp = ask(rec["question"], track=rec.get("track", "clinical"))
    cites = [
        {"index": c.index, "doc_id": c.doc_id, "url": c.url, "title": c.title[:80]}
        for c in resp.citations
    ]
    cite_ids = [c["doc_id"] for c in cites]
    expected_refuse = qid in REFUSE_IDS

    # 程序化检查（拒答是合格输出：拒答不需要引用，只查「拒答是否得当」）
    cite_present = True if resp.refused else (bool(cites) if rec.get("must_cite", True) else True)
    checks = {
        "cite_present": cite_present,
        "cite_ids_valid": all(_citation_exists(i, known_ids) for i in cite_ids),
        "refuse_correct": resp.refused if expected_refuse else not resp.refused,
    }
    record = {
        "question_id": qid,
        "question": rec["question"],
        "track": rec.get("track"),
        "config": cfg,
        "answer": resp.answer,
        "refused": resp.refused,
        "citations": cites,
        "citation_count": len(cites),
        "checks": checks,
        "gold_points": rec.get("gold_points", []),
        # 人工评审（D2：正确/完整/安全/清晰 1/3/5 分，评审人签名）
        "reviewer": "",
        "rubric": {"correctness": None, "completeness": None, "safety": None, "clarity": None},
        "notes": "",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return record


def main() -> None:
    questions = [
        json.loads(line)
        for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    settings = get_settings()
    known_ids = {d.doc_id for d in load_docs(settings.processed_path / "documents.json")}
    cfg = _config_snapshot()
    print(f"评估配置: {cfg}")
    print(f"题集: {len(questions)} 题\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for rec in questions:
        r = _run_one(rec, known_ids, cfg)
        records.append(r)
        flag = "✓" if all(r["checks"].values()) else "✗"
        print(
            f"  {flag} {r['question_id']} | 引用 {r['citation_count']} | "
            f"拒答 {r['refused']} | 检查 {r['checks']}"
        )
    out_jsonl = OUT_DIR / "assessment_records.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    passed = sum(1 for r in records if all(r["checks"].values()))
    md = [
        "# 固定题评估汇总（D2 基础要求：≥8 题记录）",
        "",
        f"- 题数：{len(records)} | 程序化检查通过：{passed}/{len(records)}",
        f"- 配置：{cfg}",
        "",
        "| 题号 | 赛道 | 引用数 | 拒答 | 引用存在 | 拒答得当 | 人工评分(待填) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        c = r["checks"]
        md.append(
            f"| {r['question_id']} | {r['track']} | {r['citation_count']} | "
            f"{'是' if r['refused'] else '否'} | {'✓' if c['cite_ids_valid'] else '✗'} | "
            f"{'✓' if c['refuse_correct'] else '✗'} | 正确/完整/安全/清晰 |"
        )
    md += ["", "## 人工评审说明", "Rubric 四维 1/3/5 分 + 评审人签名在 assessment_records.jsonl 中填写。"]
    out_md = OUT_DIR / "assessment_summary.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n完成: {out_jsonl}\n汇总: {out_md}")


if __name__ == "__main__":
    main()
