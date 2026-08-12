# -*- coding: utf-8 -*-
"""主张卡校验与 claim_scope 约束测试（不依赖网络/Chroma）。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.kb.assertions import (
    AssertionSupport,
    EvidenceAssertion,
    AssertionValidationError,
    can_support_claim_scope,
    compute_quote_hash,
    validate_and_export_assertions,
    validate_assertions_against_chunks,
)
from src.models import Chunk


def _chunk(
    *,
    doc_id: str = "pmid:111",
    text: str = "DASH diet reduces blood pressure in adults.",
    record_type: str = "published_article",
    citation_eligible: bool = True,
    source: str = "pubmed",
    url: str = "https://pubmed.ncbi.nlm.nih.gov/111/",
) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#c0",
        doc_id=doc_id,
        source=source,  # type: ignore[arg-type]
        title="t",
        text=text,
        url=url,
        evidence_level="rct",
        record_type=record_type,  # type: ignore[arg-type]
        citation_eligible=citation_eligible,
        source_locator=url or doc_id,
    )


def _assertion(
    chunk: Chunk,
    *,
    claim_scope: str = "efficacy",
    use: str = "efficacy",
    review_status: str = "needs_human_review",
    quote_hash: str | None = None,
    doc_id: str | None = None,
    chunk_id: str | None = None,
) -> EvidenceAssertion:
    return EvidenceAssertion(
        assertion_id="assertion:test-001",
        topic="dash_diet",
        claim="DASH 饮食可作为血压管理的生活方式干预方向。",
        claim_scope=claim_scope,  # type: ignore[arg-type]
        population="成人",
        intervention="DASH",
        outcome="血压",
        limitations=["不替代个体化诊疗"],
        review_status=review_status,  # type: ignore[arg-type]
        supports=[
            AssertionSupport(
                doc_id=doc_id or chunk.doc_id,
                chunk_id=chunk_id or chunk.chunk_id,
                quote_hash=quote_hash or compute_quote_hash(chunk.text),
                use=use,  # type: ignore[arg-type]
            )
        ],
    )


class AssertionValidationTests(unittest.TestCase):
    def test_valid_assertion_passes(self) -> None:
        chunk = _chunk()
        report = validate_assertions_against_chunks([_assertion(chunk)], [chunk])
        self.assertTrue(report["ok"])
        self.assertEqual(report["assertion_count"], 1)

    def test_missing_chunk_fails(self) -> None:
        chunk = _chunk()
        bad = _assertion(chunk, chunk_id="pmid:111#c9")
        report = validate_assertions_against_chunks([bad], [chunk])
        self.assertFalse(report["ok"])
        self.assertEqual(report["errors"][0]["reason"], "chunk_not_found")

    def test_doc_id_mismatch_fails(self) -> None:
        chunk = _chunk()
        bad = _assertion(chunk, doc_id="pmid:999")
        report = validate_assertions_against_chunks([bad], [chunk])
        self.assertFalse(report["ok"])
        self.assertTrue(any(e["reason"] == "doc_id_mismatch" for e in report["errors"]))

    def test_quote_hash_mismatch_fails(self) -> None:
        chunk = _chunk()
        bad = _assertion(chunk, quote_hash="sha256:deadbeef")
        report = validate_assertions_against_chunks([bad], [chunk])
        self.assertFalse(report["ok"])
        self.assertTrue(any(e["reason"] == "quote_hash_mismatch" for e in report["errors"]))

    def test_trial_registry_cannot_support_efficacy(self) -> None:
        chunk = _chunk(
            doc_id="nct:NCT00000001",
            source="clinicaltrials",
            record_type="trial_registry",
            citation_eligible=True,
            url="https://clinicaltrials.gov/study/NCT00000001",
            text="Status: COMPLETED. Primary Outcome: BP change.",
        )
        report = validate_assertions_against_chunks([_assertion(chunk)], [chunk])
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("scope_not_allowed:efficacy" in e["reason"] for e in report["errors"])
        )

    def test_trial_registry_can_support_protocol(self) -> None:
        chunk = _chunk(
            doc_id="nct:NCT00000001",
            source="clinicaltrials",
            record_type="trial_registry",
            citation_eligible=False,
            url="https://clinicaltrials.gov/study/NCT00000001",
            text="Status: RECRUITING. Primary Outcome: BP change.",
        )
        assertion = _assertion(chunk, claim_scope="protocol", use="protocol")
        report = validate_assertions_against_chunks([assertion], [chunk])
        self.assertTrue(report["ok"])

    def test_can_support_claim_scope_matrix(self) -> None:
        pub = {
            "record_type": "published_article",
            "citation_eligible": True,
            "source": "pubmed",
        }
        reg = {
            "record_type": "trial_registry",
            "citation_eligible": True,
            "source": "clinicaltrials",
        }
        self.assertTrue(can_support_claim_scope(pub, "efficacy"))
        self.assertFalse(can_support_claim_scope(reg, "efficacy"))
        self.assertFalse(can_support_claim_scope(reg, "risk"))
        self.assertTrue(can_support_claim_scope(reg, "protocol"))
        self.assertTrue(can_support_claim_scope(reg, "recruitment_status"))

    def test_export_disabled_when_curated_missing(self) -> None:
        chunk = _chunk()
        with tempfile.TemporaryDirectory() as tmp:
            report = validate_and_export_assertions(
                [chunk],
                curated_path=Path(tmp) / "missing.jsonl",
                processed_dir=Path(tmp) / "processed",
            )
            self.assertTrue(report["disabled"])
            self.assertTrue(report["ok"])

    def test_export_fails_hard_on_invalid(self) -> None:
        chunk = _chunk()
        with tempfile.TemporaryDirectory() as tmp:
            curated = Path(tmp) / "assertions.jsonl"
            bad = _assertion(chunk, quote_hash="sha256:bad")
            curated.write_text(bad.model_dump_json() + "\n", encoding="utf-8")
            with self.assertRaises(AssertionValidationError):
                validate_and_export_assertions(
                    [chunk],
                    curated_path=curated,
                    processed_dir=Path(tmp) / "processed",
                )


if __name__ == "__main__":
    unittest.main()
