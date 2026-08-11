# -*- coding: utf-8 -*-
"""
B 组功能自检脚本：逐个验证新补全的函数，输出 PASS/FAIL。

运行：
    python scripts/self_check_b.py
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_hybrid() -> None:
    from src.retrieval.hybrid import (
        diversify_by_source,
        explain_retrieval,
        filter_by_year_range,
        reciprocal_rank_fusion,
        score_evidence_priority,
    )

    assert score_evidence_priority({"evidence_level": "guideline", "year": 2024}) > 1.0
    items = [
        {"chunk_id": "a", "year": 2020},
        {"chunk_id": "b", "year": -1},
        {"chunk_id": "c"},
    ]
    assert [x["chunk_id"] for x in filter_by_year_range(items)] == ["a", "b", "c"]
    assert [x["chunk_id"] for x in filter_by_year_range(items, 2019)] == ["a"]
    expl = explain_retrieval(
        "q", [{"source": "pubmed", "title": "A", "evidence_level": "rct"}]
    )
    assert sorted(expl) == ["notes", "query", "sources", "why_selected"]
    div = diversify_by_source(
        [
            {"chunk_id": "a", "source": "pubmed"},
            {"chunk_id": "b", "source": "pubmed"},
            {"chunk_id": "c", "source": "pubmed"},
            {"chunk_id": "d", "source": "wiki"},
        ]
    )
    assert [x["chunk_id"] for x in div] == ["a", "b", "d"]
    rrf = reciprocal_rank_fusion(
        [{"chunk_id": "a"}, {"chunk_id": "b"}],
        [{"chunk_id": "b"}, {"chunk_id": "c"}],
    )
    assert [x["chunk_id"] for x in rrf] == ["b", "a", "c"]


def test_generation() -> None:
    from src.generation.answer import (
        compute_faithfulness_proxy,
        enforce_citation_density,
        format_reference_section,
    )
    from src.models import Citation

    answer = "生活方式干预有证据支持[1]，但个体化剂量应咨询医生。"
    contexts = [
        {"text": "生活方式干预与药物管理均有证据支持。", "title": "研究A"}
    ]
    score = compute_faithfulness_proxy(answer, contexts)
    assert 0.0 < score <= 1.0
    assert enforce_citation_density(answer, 1) is True
    assert enforce_citation_density(answer, 2) is False
    refs = format_reference_section(
        [
            Citation(
                index=1,
                doc_id="pmid:1",
                title="A",
                source="pubmed",
                year=2020,
                url="https://x",
                evidence_level="rct",
                snippet="s",
            )
        ]
    )
    assert "参考文献" in refs


def test_cite_check() -> None:
    from src.tools.cite_check import (
        detect_unsupported_claims,
        repair_answer_with_valid_cites,
        verify_citations,
    )

    contexts = [{"text": "生活方式干预与药物管理均有证据支持。", "title": "研究A"}]
    bad = "结果来自PMID: 99999 的研究[1][9]。"
    check = verify_citations(bad, contexts)
    assert check["fake_pmids"] == ["99999"]
    assert check["invalid_brackets"] == [9]
    repaired = repair_answer_with_valid_cites(bad, contexts, check)
    assert "PMID" not in repaired
    assert "[9]" not in repaired
    claims = detect_unsupported_claims(
        "这是没有引用的结论。另一句有依据[1]。", contexts
    )
    assert len(claims) == 1


def test_live_search() -> None:
    from src.models import EvidenceDoc
    from src.tools.live_search import (
        merge_live_and_offline_docs,
        should_trigger_live_search,
    )

    offline = [EvidenceDoc(doc_id="local:1", source="local", title="t1", text="x")]
    live = [
        EvidenceDoc(doc_id="local:1", source="local", title="t1", text="x"),
        EvidenceDoc(doc_id="pubmed:2", source="pubmed", title="t2", text="y"),
    ]
    merged = merge_live_and_offline_docs(offline, live, max_total=2)
    assert [d.doc_id for d in merged] == ["local:1", "pubmed:2"]
    assert should_trigger_live_search("q", 1) is True
    assert should_trigger_live_search("q", 5) is False


def test_config() -> None:
    from src.config import validate_runtime_config

    result = validate_runtime_config()
    assert set(["ok", "offline_mode", "issues"]).issubset(result)
    assert isinstance(result["ok"], bool)


def test_tracks() -> None:
    from src.models import AskResponse
    from src.tracks.clinical import (
        build_clinical_answer_outline,
        rank_contexts_for_clinical,
    )
    from src.tracks.nutrition import (
        build_nutrition_action_tips,
        simplify_medical_terms,
    )
    from src.tracks.pipeline import (
        attach_retrieval_explanation,
        detect_track_from_question,
    )

    contexts = [
        {
            "chunk_id": "a",
            "doc_id": "d1",
            "title": "观察研究",
            "evidence_level": "observational",
            "year": 2020,
            "source": "pubmed",
            "text": "x",
            "tags": ["diet"],
        },
        {
            "chunk_id": "b",
            "doc_id": "d2",
            "title": "RCT 试验",
            "evidence_level": "rct",
            "year": 2022,
            "source": "pubmed",
            "text": "y",
            "tags": ["diet"],
        },
    ]
    assert [c["chunk_id"] for c in rank_contexts_for_clinical(contexts)] == ["b", "a"]
    # 强制离线回退，避免自检时调用真实 LLM 导致不稳定
    import src.tracks.clinical as clinical_mod

    original_llm = clinical_mod.get_llm
    clinical_mod.get_llm = lambda: types.SimpleNamespace(is_offline=True)
    try:
        outline = build_clinical_answer_outline(contexts)
    finally:
        clinical_mod.get_llm = original_llm
    assert sorted(outline) == ["conclusion", "evidence_levels", "key_studies", "limitations"]
    assert "rct" in outline["evidence_levels"]
    assert outline["key_studies"]
    tips = build_nutrition_action_tips(
        [
            {"text": "地中海饮食可降低心血管风险。", "title": "x", "tags": ["mediterranean"]},
            {"text": "限钠可降血压。", "title": "y", "tags": "hypertension,diet"},
        ]
    )
    assert 3 <= len(tips) <= 5
    assert "坏胆固醇" in simplify_medical_terms("LDL-C 与高血压相关")
    assert detect_track_from_question("地中海饮食能降血压吗") == "nutrition"
    assert detect_track_from_question("他汀如何降血脂") == "clinical"
    resp = AskResponse(answer="a", citation_check={"ok": True})
    attached = attach_retrieval_explanation(
        resp, {"query": "q", "why_selected": ["x"], "sources": ["pubmed"], "notes": "n"}
    )
    assert attached.citation_check["explanation"]["query"] == "q"
    assert "explanation" not in resp.citation_check


def test_eval_bench() -> None:
    from src.tracks.eval_bench import (
        compare_metric_delta,
        export_human_rubric_template,
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
        {
            "id": "n1",
            "track": "nutrition",
            "question": "q3",
            "rag": {"fake_citation_count": 0, "refused": False, "n_contexts": 4, "gold_coverage": 0.8},
            "baseline": {"fake_citation_signal": 0, "gold_coverage": 0.3},
        },
    ]
    cases = pick_typical_cases(results, n=3)
    assert 1 <= len(cases) <= 3
    with tempfile.TemporaryDirectory() as td:
        md = export_human_rubric_template(Path(td) / "rubric.md")
        csvf = export_human_rubric_template(Path(td) / "rubric.csv")
        assert md.exists() and csvf.exists()
    delta = compare_metric_delta(
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
    assert delta["fake_citation_delta"] == -0.5
    assert delta["gold_coverage_delta"] == 0.4


def test_llm_helpers() -> None:
    import src.llm as llm_mod

    class FakeLLM:
        def chat(self, messages, **kwargs):
            return '```json\n{"ok": true}\n```'

        def embed(self, texts):
            return [[float(i + 1)] for i in range(len(texts))]

    original = llm_mod.get_llm
    llm_mod.get_llm = lambda: FakeLLM()
    try:
        assert llm_mod.with_json_mode_chat([{"role": "user", "content": "x"}]) == {"ok": True}
        calls = {"n": 0}

        class CountingEmbed:
            def embed(self, texts):
                calls["n"] += 1
                return [[float(i + 1)] for i in range(len(texts))]

        llm_mod.get_llm = lambda: CountingEmbed()
        with tempfile.TemporaryDirectory() as td:
            v1 = llm_mod.embed_with_cache(["a", "b"], cache_dir=td)
            v2 = llm_mod.embed_with_cache(["a", "b"], cache_dir=td)
            assert v1 == v2
            assert calls["n"] == 1
    finally:
        llm_mod.get_llm = original


def main() -> None:
    checks = [
        ("hybrid 检索增强", test_hybrid),
        ("generation 生成质量", test_generation),
        ("cite_check 引用校验", test_cite_check),
        ("live_search 在线补检索", test_live_search),
        ("config 运行时自检", test_config),
        ("tracks 赛道功能", test_tracks),
        ("eval_bench 评测工具", test_eval_bench),
        ("llm JSON/缓存", test_llm_helpers),
    ]
    failed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    if failed:
        print(f"\n自检未全部通过，失败 {failed} 项")
        raise SystemExit(1)
    print("\nB 组自检全部通过")


if __name__ == "__main__":
    main()
