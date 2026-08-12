# -*- coding: utf-8 -*-
"""自适应最小证据集与充分性状态测试（不依赖网络/Chroma）。"""

from __future__ import annotations

import unittest

from src.generation.answer import generate_answer
from src.kb.assertions import AssertionSupport, EvidenceAssertion, compute_quote_hash
from src.retrieval.evidence_set import (
    build_evidence_plan,
    identify_evidence_units,
    render_allowed_claims_answer,
)
from src.tracks.prompt_profiles import build_synthesis_messages


def _ctx(
    doc_id: str,
    *,
    text: str = "evidence text",
    record_type: str = "published_article",
    citation_eligible: bool = True,
    evidence_level: str = "rct",
    source: str = "pubmed",
    chunk_suffix: str = "c0",
) -> dict:
    return {
        "chunk_id": f"{doc_id}#{chunk_suffix}",
        "doc_id": doc_id,
        "title": f"title-{doc_id}",
        "text": text,
        "source": source,
        "url": f"https://example.org/{doc_id}",
        "evidence_level": evidence_level,
        "record_type": record_type,
        "citation_eligible": citation_eligible,
        "year": 2020,
    }


def _reviewed(
    assertion_id: str,
    topic: str,
    claim: str,
    claim_scope: str,
    ctx: dict,
) -> EvidenceAssertion:
    return EvidenceAssertion(
        assertion_id=assertion_id,
        topic=topic,
        claim=claim,
        claim_scope=claim_scope,  # type: ignore[arg-type]
        population="成人",
        intervention="x",
        outcome="y",
        limitations=["局限A"],
        review_status="human_reviewed",
        reviewed_at="2026-08-01",
        supports=[
            AssertionSupport(
                doc_id=ctx["doc_id"],
                chunk_id=ctx["chunk_id"],
                quote_hash=compute_quote_hash(ctx["text"]),
                use=claim_scope,  # type: ignore[arg-type]
            )
        ],
    )


class EvidenceSetTests(unittest.TestCase):
    def test_duplicate_chunks_same_doc_do_not_fill_set(self) -> None:
        base_text = "DASH diet lowers blood pressure"
        c0 = _ctx("pmid:1", text=base_text, chunk_suffix="c0")
        c1 = _ctx("pmid:1", text=base_text + " more", chunk_suffix="c1")
        c2 = _ctx("pmid:1", text=base_text + " again", chunk_suffix="c2")
        other = _ctx("pmid:2", text="risk profile of DASH", evidence_level="observational")
        assertions = [
            _reviewed(
                "assertion:dash-e",
                "dash_diet",
                "DASH 可用于血压管理。",
                "efficacy",
                c0,
            ),
            _reviewed(
                "assertion:dash-r",
                "dash_diet",
                "DASH 总体安全性可接受，但仍需关注个体差异。",
                "risk",
                other,
            ),
        ]
        candidates = [c0, c1, c2, other]
        plan = build_evidence_plan(
            "DASH 饮食是否有效？有什么风险？",
            initial_candidates=candidates,
            assertions=assertions,
            max_evidence=5,
            max_rounds=0,
        )
        doc_ids = [c["doc_id"] for c in plan.selected_contexts]
        self.assertEqual(doc_ids.count("pmid:1"), 1)
        self.assertIn("pmid:2", doc_ids)

    def test_partial_when_only_one_of_two_required_units(self) -> None:
        eff = _ctx("pmid:10", text="efficacy evidence for DASH")
        assertions = [
            _reviewed(
                "assertion:only-eff",
                "dash_diet",
                "DASH 可作为降压生活方式方向。",
                "efficacy",
                eff,
            )
        ]
        plan = build_evidence_plan(
            "DASH 饮食是否有效？有什么副作用风险？",
            initial_candidates=[eff],
            assertions=assertions,
            max_evidence=5,
            max_rounds=0,
        )
        self.assertEqual(plan.status, "partial")
        self.assertTrue(plan.allowed_claims)
        self.assertTrue(any("risk" in m for m in plan.missing_evidence))

    def test_insufficient_without_qualified_primary_support(self) -> None:
        registry = _ctx(
            "nct:NCT1",
            text="Status RECRUITING primary outcome BP",
            record_type="trial_registry",
            citation_eligible=False,
            source="clinicaltrials",
            evidence_level="other",
        )
        # 故意构造一张 efficacy 卡绑定注册记录——选择器应拒绝覆盖
        bad = EvidenceAssertion(
            assertion_id="assertion:bad-reg",
            topic="dash_diet",
            claim="假疗效",
            claim_scope="efficacy",
            limitations=["x"],
            review_status="human_reviewed",
            supports=[
                AssertionSupport(
                    doc_id=registry["doc_id"],
                    chunk_id=registry["chunk_id"],
                    quote_hash=compute_quote_hash(registry["text"]),
                    use="efficacy",
                )
            ],
        )
        plan = build_evidence_plan(
            "DASH 饮食是否有效降低血压？",
            initial_candidates=[registry],
            assertions=[bad],
            max_evidence=5,
            max_rounds=0,
        )
        self.assertEqual(plan.status, "insufficient")
        self.assertFalse(plan.allowed_claims)

    def test_unmapped_not_marked_sufficient(self) -> None:
        ctx = _ctx("pmid:99", text="unrelated")
        assertions = [
            _reviewed(
                "assertion:dash",
                "dash_diet",
                "DASH claim",
                "efficacy",
                ctx,
            )
        ]
        plan = build_evidence_plan(
            "火星尘埃能否治疗高血压？",
            initial_candidates=[ctx],
            assertions=assertions,
            max_evidence=5,
            max_rounds=0,
        )
        self.assertEqual(plan.status, "unmapped")
        self.assertNotEqual(plan.status, "sufficient")

    def test_partial_generation_uses_allowed_claims_and_gaps(self) -> None:
        claim = "DASH 可作为成人血压管理的生活方式干预方向。"
        messages = build_synthesis_messages(
            "clinical",
            "DASH 是否有效？风险如何？",
            "[1] evidence",
            allowed_claims=[claim],
            missing_evidence=["未覆盖风险单元"],
            limitations=["不替代个体化诊疗"],
            evidence_status="partial",
        )
        system = messages[0]["content"]
        self.assertIn(claim, system)
        self.assertIn("未覆盖风险单元", system)
        self.assertIn("ALLOWED CLAIMS", system)

        plan_like_answer = render_allowed_claims_answer(
            build_evidence_plan(
                "DASH 饮食是否有效？有什么风险？",
                initial_candidates=[_ctx("pmid:10", text="DASH efficacy")],
                assertions=[
                    _reviewed(
                        "assertion:e",
                        "dash_diet",
                        claim,
                        "efficacy",
                        _ctx("pmid:10", text="DASH efficacy"),
                    )
                ],
                max_rounds=0,
            )
        )
        self.assertIn(claim, plan_like_answer)
        self.assertTrue("缺口" in plan_like_answer or "risk" in plan_like_answer.lower() or "未覆盖" in plan_like_answer)

    def test_generate_answer_partial_includes_gap(self) -> None:
        contexts = [
            _ctx(
                "pmid:10",
                text="DASH diet blood pressure trial evidence",
            )
        ]
        answer, _citations, refused = generate_answer(
            "DASH 是否有效？风险如何？",
            contexts,
            system_persona="tester",
            answer_style="简洁",
            track="clinical",
            allowed_claims=["DASH 可作为血压管理方向。"],
            limitations=["局限"],
            missing_evidence=["未覆盖风险证据"],
            evidence_status="partial",
            force_evidence_card=True,
        )
        self.assertFalse(refused)
        self.assertIn("DASH 可作为血压管理方向。", answer)
        self.assertIn("未覆盖风险证据", answer)

    def test_identify_units_for_efficacy_and_risk(self) -> None:
        units = identify_evidence_units("DASH 是否有效？有什么风险？", topic="dash_diet")
        ids = {u.unit_id for u in units}
        self.assertIn("efficacy", ids)
        self.assertIn("risk", ids)


if __name__ == "__main__":
    unittest.main()
