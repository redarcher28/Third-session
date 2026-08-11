# -*- coding: utf-8 -*-
"""
B 组交互式测试菜单。

运行：
    python scripts/interactive_check_b.py

输入菜单数字即可逐项测试；涉及文本的选项会先提示输入，
直接回车则使用内置示例。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import self_check_b


def _ask_text(prompt: str, default: str) -> str:
    value = input(f"{prompt}（直接回车用示例）\n> ").strip()
    return value or default


def run_ask() -> None:
    from src.tracks.pipeline import ask

    question = _ask_text(
        "输入要测试的问题",
        "高血压患者为什么有时要长期吃药？有哪些指南或研究依据？",
    )
    track = input("赛道 clinical / nutrition（直接回车 clinical）\n> ").strip() or "clinical"
    if track not in ("clinical", "nutrition"):
        track = "clinical"
    raw_from = input("证据起始年份（直接回车不限）\n> ").strip()
    raw_to = input("证据截止年份（直接回车不限）\n> ").strip()
    year_from = int(raw_from) if raw_from.isdigit() else None
    year_to = int(raw_to) if raw_to.isdigit() else None
    resp = ask(question, track=track, year_from=year_from, year_to=year_to)
    print("\n===== ask 结果 =====")
    print(f"track: {resp.track}")
    print(f"refused: {resp.refused}")
    print(f"rewritten_query: {resp.rewritten_query}")
    print(f"contexts: {len(resp.contexts)}")
    print(f"citation_check: {resp.citation_check}")
    print(f"answer:\n{resp.answer[:600]}")
    print("====================\n")


def run_generation() -> None:
    from src.generation.answer import (
        compute_faithfulness_proxy,
        enforce_citation_density,
        format_reference_section,
    )
    from src.models import Citation

    answer = _ask_text(
        "输入一段回答（可含 [n] 引用）",
        "生活方式干预有证据支持[1]，但个体化剂量应咨询医生。",
    )
    contexts = [{"text": "生活方式干预与药物管理均有证据支持。", "title": "研究A"}]
    print(f"\nfaithfulness: {compute_faithfulness_proxy(answer, contexts)}")
    print(f"citation density(>=2): {enforce_citation_density(answer, 2)}")
    print(
        "参考文献示例:\n"
        + format_reference_section(
            [
                Citation(
                    index=1,
                    doc_id="pmid:1",
                    title="研究A",
                    source="pubmed",
                    year=2020,
                    url="https://x",
                    evidence_level="rct",
                    snippet="s",
                )
            ]
        )
    )


def run_cite_check() -> None:
    from src.tools.cite_check import (
        detect_unsupported_claims,
        repair_answer_with_valid_cites,
        verify_citations,
    )

    answer = _ask_text(
        "输入一段可能含假引用的回答",
        "结果来自PMID: 99999 的研究[1][9]。",
    )
    contexts = [{"text": "生活方式干预与药物管理均有证据支持。", "title": "研究A"}]
    check = verify_citations(answer, contexts)
    print(f"\n校验结果: {check}")
    print(f"修复后: {repair_answer_with_valid_cites(answer, contexts, check)}")
    print(f"可疑无引用句: {detect_unsupported_claims(answer, contexts)}")


def run_live_search() -> None:
    from src.tools.live_search import should_trigger_live_search

    question = _ask_text("输入用户问题", "最近有没有高血压治疗的新指南？")
    raw = input("离线命中条数（直接回车 2）\n> ").strip() or "2"
    try:
        hits = int(raw)
    except ValueError:
        hits = 2
    print(f"\nshould_trigger_live_search: {should_trigger_live_search(question, hits)}")


def run_config() -> None:
    from src.config import validate_runtime_config

    print(f"\n运行时自检: {validate_runtime_config()}\n")


def run_track_detect() -> None:
    from src.tracks.clinical import build_clinical_answer_outline
    from src.tracks.nutrition import build_nutrition_action_tips
    from src.tracks.pipeline import detect_track_from_question

    question = _ask_text("输入用户问题", "地中海饮食对心血管风险有什么证据？")
    track = detect_track_from_question(question)
    print(f"\n识别赛道: {track}")
    contexts = [
        {
            "chunk_id": "a",
            "doc_id": "d1",
            "title": "RCT 试验",
            "evidence_level": "rct",
            "year": 2022,
            "source": "pubmed",
            "text": "地中海饮食可降低心血管风险。",
            "tags": ["mediterranean"],
        },
        {
            "chunk_id": "b",
            "doc_id": "d2",
            "title": "观察研究",
            "evidence_level": "observational",
            "year": 2020,
            "source": "pubmed",
            "text": "限钠可降低血压。",
            "tags": ["hypertension"],
        },
    ]
    if track == "nutrition":
        print(f"行动建议: {build_nutrition_action_tips(contexts)}")
    else:
        print(f"临床大纲: {build_clinical_answer_outline(contexts)}")


def run_simplify() -> None:
    from src.tracks.nutrition import simplify_medical_terms

    text = _ask_text(
        "输入一段带术语的文本",
        "LDL-C 与甘油三酯升高会增加心血管事件风险，指南建议他汀类药物治疗。",
    )
    print(f"\n通俗化: {simplify_medical_terms(text)}\n")


def run_eval_tools() -> None:
    from src.tracks.eval_bench import (
        compare_metric_delta,
        pick_typical_cases,
    )

    results = [
        {
            "id": "c1",
            "track": "clinical",
            "question": "q1",
            "rag": {"fake_citation_count": 1, "refused": False, "n_contexts": 3, "gold_coverage": 0.2},
            "baseline": {"fake_citation_signal": 2, "gold_coverage": 0.5},
        },
        {
            "id": "c2",
            "track": "clinical",
            "question": "q2",
            "rag": {"fake_citation_count": 0, "refused": True, "n_contexts": 0, "gold_coverage": 0.0},
            "baseline": {"fake_citation_signal": 0, "gold_coverage": 0.8},
        },
    ]
    print(f"\n典型 Case: {[c['id'] for c in pick_typical_cases(results, n=2)]}")
    print(
        "指标差值: "
        + str(
            compare_metric_delta(
                {
                    "rag_fake_citation_rate": 0.1,
                    "baseline_fake_citation_signal_rate": 0.6,
                    "rag_avg_gold_coverage": 0.7,
                    "baseline_avg_gold_coverage": 0.3,
                    "rag_refusal_rate": 0.1,
                    "rag_citation_coverage": 0.9,
                    "rag_empty_context_cases": 1,
                }
            )
        )
    )


def main() -> None:
    menu = {
        "1": ("完整问答 ask()", run_ask),
        "2": ("混合检索增强（示例断言）", self_check_b.test_hybrid),
        "3": ("生成质量（输入回答）", run_generation),
        "4": ("引用校验与修复（输入回答）", run_cite_check),
        "5": ("在线补检索判断（输入问题）", run_live_search),
        "6": ("配置自检", run_config),
        "7": ("赛道识别 + 临床大纲/营养建议（输入问题）", run_track_detect),
        "8": ("营养术语简化（输入文本）", run_simplify),
        "9": ("评测工具（示例）", run_eval_tools),
        "0": ("退出", None),
    }
    print("===== B 组交互式测试 =====")
    while True:
        print("\n菜单：")
        for key, (label, _) in menu.items():
            print(f"  {key}) {label}")
        choice = input("请选择: ").strip()
        if choice == "0":
            print("已退出")
            return
        item = menu.get(choice)
        if item is None or item[1] is None:
            print("无效选项，请重试")
            continue
        print(f"\n--- {item[0]} ---")
        try:
            result = item[1]()
            if result is None:
                print("通过")
        except Exception as e:
            print(f"执行失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
