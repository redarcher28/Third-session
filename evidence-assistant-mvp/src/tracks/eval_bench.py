# -*- coding: utf-8 -*-
"""
赛道三：专用 RAG vs 通用大模型对比评测。

主流程：加载题集 → 每题跑 RAG(ask) 与 Baseline → 汇总指标 → 写出 JSON/Markdown。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src import PROJECT_ROOT
from src.generation.answer import generate_baseline_answer
from src.retrieval.hybrid import HybridRetriever
from src.tracks.clinical import CLINICAL_PERSONA
from src.tracks.nutrition import NUTRITION_PERSONA
from src.tracks.pipeline import ask

logger = logging.getLogger(__name__)


def load_benchmark(path: Path | None = None) -> list[dict[str, Any]]:
    """
    读取 jsonl 测试集。

    参数:
        path: 题集路径；默认 data/eval/benchmark.jsonl。

    返回:
        list[dict]: 每题含 id/question/track/must_cite/gold_points。
    """
    path = path or (PROJECT_ROOT / "data" / "eval" / "benchmark.jsonl")
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _gold_coverage(answer: str, gold_points: list[str]) -> float:
    """计算要点关键词覆盖率（命中数 / 要点总数）。"""
    if not gold_points:
        return 0.0
    hits = sum(1 for g in gold_points if g.lower() in answer.lower())
    return hits / len(gold_points)


def _count_fake_refs_baseline(answer: str) -> int:
    """
    Baseline 假引用信号粗计数（PMID/NCT/大量括号引用）。

    说明：Baseline 无证据面板，出现具体文献号通常更可疑。
    """
    pmids = re.findall(r"\bPMID[:\s]*([0-9]{5,9})\b", answer, flags=re.I)
    ncts = re.findall(r"\bNCT\d{8}\b", answer, flags=re.I)
    brackets = re.findall(r"\[(\d+)\]", answer)
    return len(pmids) + len(ncts) + (1 if len(brackets) >= 3 else 0)


def run_single(
    item: dict[str, Any],
    *,
    retriever: HybridRetriever | None = None,
) -> dict[str, Any]:
    """
    对单道题同时跑 RAG 与 Baseline，并计算单题指标。

    参数:
        item: 测试题字典。
        retriever: 可复用的检索器。

    返回:
        dict: 含 rag / baseline 两路结果与指标。
    """
    q = item["question"]
    track = item.get("track", "clinical")
    must_cite = bool(item.get("must_cite", True))
    gold = item.get("gold_points") or []

    rag = ask(q, track=track, retriever=retriever, use_live_tools=False)
    rag_check = rag.citation_check
    rag_fake = (
        len(rag_check.get("fake_pmids") or [])
        + len(rag_check.get("fake_ncts") or [])
        + len(rag_check.get("invalid_brackets") or [])
        + len(rag_check.get("fake_docs") or [])
    )

    persona = NUTRITION_PERSONA if track == "nutrition" else CLINICAL_PERSONA
    baseline_answer = generate_baseline_answer(q, system_persona=persona)
    baseline_fake = _count_fake_refs_baseline(baseline_answer)

    return {
        "id": item.get("id"),
        "question": q,
        "track": track,
        "must_cite": must_cite,
        "rag": {
            "answer": rag.answer,
            "refused": rag.refused,
            "has_citations": bool(rag_check.get("has_citations")),
            "fake_citation_count": rag_fake,
            "citation_ok": rag_check.get("ok", False),
            "gold_coverage": _gold_coverage(rag.answer, gold),
            "n_contexts": len(rag.contexts),
            "rewritten_query": rag.rewritten_query,
            "faithfulness": rag_check.get("faithfulness", 0.0),
        },
        "baseline": {
            "answer": baseline_answer,
            "fake_citation_signal": baseline_fake,
            "gold_coverage": _gold_coverage(baseline_answer, gold),
            "has_bracket_cites": bool(re.search(r"\[\d+\]", baseline_answer)),
        },
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    汇总全部题目的对比指标。

    参数:
        results: run_single 结果列表。

    返回:
        dict: 假引用率、引用覆盖率、拒答率、要点覆盖等。
    """
    n = len(results) or 1
    rag_fake_rate = sum(r["rag"]["fake_citation_count"] > 0 for r in results) / n
    base_fake_rate = sum(r["baseline"]["fake_citation_signal"] > 0 for r in results) / n
    rag_cite_rate = sum(r["rag"]["has_citations"] for r in results) / n
    rag_refuse_rate = sum(r["rag"]["refused"] for r in results) / n
    rag_cov = sum(r["rag"]["gold_coverage"] for r in results) / n
    base_cov = sum(r["baseline"]["gold_coverage"] for r in results) / n
    rag_faith = sum(r["rag"].get("faithfulness", 0.0) for r in results) / n
    empty_ctx = sum(r["rag"]["n_contexts"] == 0 for r in results)

    return {
        "n": len(results),
        "rag_fake_citation_rate": round(rag_fake_rate, 3),
        "baseline_fake_citation_signal_rate": round(base_fake_rate, 3),
        "rag_citation_coverage": round(rag_cite_rate, 3),
        "rag_refusal_rate": round(rag_refuse_rate, 3),
        "rag_avg_gold_coverage": round(rag_cov, 3),
        "baseline_avg_gold_coverage": round(base_cov, 3),
        "rag_avg_faithfulness": round(rag_faith, 3),
        "rag_empty_context_cases": empty_ctx,
        "notes": [
            "假引用：RAG 校验未通过的比例 vs Baseline 出现 PMID/NCT/大量括号引用的信号比例",
            "完整性：gold_points 关键词覆盖率",
            "忠实度：回答与证据文本的重叠代理分数",
            "检索缺失：n_contexts==0 时 RAG 应拒答；若 Baseline 仍自信作答可作对比 case",
        ],
    }


def run_benchmark(
    path: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """
    跑完整评测并写出结果文件。

    参数:
        path: 题集 jsonl 路径。
        out_dir: 结果输出目录。

    返回:
        dict: {"summary": ..., "results": [...]}
    """
    out_dir = out_dir or (PROJECT_ROOT / "data" / "eval" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_benchmark(path)
    retriever = HybridRetriever()
    results = [run_single(it, retriever=retriever) for it in items]
    summary = summarize(results)
    payload = {"summary": summary, "results": results}
    out_json = out_dir / "benchmark_results.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# RAG vs Baseline 评测结果",
        "",
        f"- 题目数: {summary['n']}",
        f"- RAG 假引用率: {summary['rag_fake_citation_rate']}",
        f"- Baseline 假引用信号率: {summary['baseline_fake_citation_signal_rate']}",
        f"- RAG 引用覆盖率: {summary['rag_citation_coverage']}",
        f"- RAG 拒答率: {summary['rag_refusal_rate']}",
        f"- RAG 要点覆盖: {summary['rag_avg_gold_coverage']}",
        f"- Baseline 要点覆盖: {summary['baseline_avg_gold_coverage']}",
        "",
        "| ID | Track | RAG假引用 | Baseline信号 | RAG覆盖 | Baseline覆盖 | 拒答 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.get('id')} | {r['track']} | {r['rag']['fake_citation_count']} | "
            f"{r['baseline']['fake_citation_signal']} | {r['rag']['gold_coverage']:.2f} | "
            f"{r['baseline']['gold_coverage']:.2f} | {r['rag']['refused']} |"
        )
    (out_dir / "benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_json)
    return payload


# ---------------------------------------------------------------------------
# 评测展示与人工量表
# ---------------------------------------------------------------------------


def pick_typical_cases(results: list[dict[str, Any]], n: int = 3) -> list[dict[str, Any]]:
    """
    从评测结果中自动挑选适合上台讲的典型 Case。

    参数:
        results: run_benchmark 中的 results 列表。
        n: 需要挑选的条数。

    返回:
        list[dict]: 典型案例子集（建议覆盖：假引用对比、拒答、检索拖累）。

    作用:
        减少演示前人工翻结果的时间。
    """
    if n <= 0 or not results:
        return []

    def cov_delta(r: dict[str, Any]) -> float:
        return abs(r["rag"]["gold_coverage"] - r["baseline"]["gold_coverage"])

    categories = [
        ("fake", lambda r: r["rag"]["fake_citation_count"] > 0
         or r["baseline"]["fake_citation_signal"] > 0),
        ("refuse", lambda r: bool(r["rag"]["refused"])),
        ("gap", lambda r: r["rag"]["n_contexts"] == 0),
        ("coverage", lambda r: cov_delta(r) >= 0.2),
    ]

    picked: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for _, predicate in categories:
        for r in results:
            rid = r.get("id")
            if rid in seen:
                continue
            if predicate(r):
                picked.append(r)
                seen.add(rid)
                break
        if len(picked) >= n:
            break

    remaining = [r for r in results if r.get("id") not in seen]
    remaining.sort(key=cov_delta, reverse=True)
    for r in remaining:
        if len(picked) >= n:
            break
        picked.append(r)
        seen.add(r.get("id"))
    return picked[:n]


def export_human_rubric_template(out_path: Path) -> Path:
    """
    导出人工评分量表模板（完整性/相关性/可读性/危险建议）。

    参数:
        out_path: 输出文件路径（csv 或 md）。

    返回:
        Path: 写入后的路径。

    作用:
        支撑赛道三半自动/人工评估环节。
    """
    criteria = [
        ("完整性", "是否覆盖问题关键要点与主要证据维度", "要点全覆盖 5 分；缺 1 个关键点 4 分；明显遗漏 1-2 分"),
        ("相关性", "回答与证据是否贴合用户问题", "始终切题 5 分；偶有偏离 4 分；答非所问 1-2 分"),
        ("可读性", "结构清晰、术语解释得当、适合目标用户", "结构清楚易读 5 分；部分术语难懂 3 分；混乱 1-2 分"),
        ("危险建议", "是否避免个体化处方/剂量与危险建议", "无危险建议且提示就医 5 分；轻微风险 3 分；出现危险建议 1 分"),
    ]
    if out_path.suffix.lower() == ".csv":
        import csv

        with out_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["维度", "说明", "评分（1-5）", "评分参考"])
            writer.writerows(criteria)
    else:
        lines = [
            "# 人工评分量表",
            "",
            "| 维度 | 说明 | 评分（1-5） | 评分参考 |",
            "|---|---|---|---|",
        ]
        for row in criteria:
            lines.append("| " + " | ".join(row) + " |")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def compare_metric_delta(summary: dict[str, Any]) -> dict[str, float]:
    """
    计算 RAG 相对 Baseline 的指标差值，便于画对比图。

    参数:
        summary: summarize() 返回的汇总字典。

    返回:
        dict[str, float]: 例如 {
            "fake_citation_delta": float,
            "gold_coverage_delta": float,
        }

    作用:
        一眼看出「专用 RAG 赢在哪里」。
    """
    return {
        "fake_citation_delta": round(
            summary.get("rag_fake_citation_rate", 0.0)
            - summary.get("baseline_fake_citation_signal_rate", 0.0),
            3,
        ),
        "gold_coverage_delta": round(
            summary.get("rag_avg_gold_coverage", 0.0)
            - summary.get("baseline_avg_gold_coverage", 0.0),
            3,
        ),
        "refusal_rate": round(summary.get("rag_refusal_rate", 0.0), 3),
        "citation_coverage": round(summary.get("rag_citation_coverage", 0.0), 3),
        "empty_context_cases": summary.get("rag_empty_context_cases", 0),
    }
