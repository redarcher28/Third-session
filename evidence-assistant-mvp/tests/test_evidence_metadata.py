# -*- coding: utf-8 -*-
"""证据元数据：试验注册分级、切块传播、检索兜底。"""

from __future__ import annotations

import unittest

from src.ingest import (
    classify_trial_registry,
    normalize_evidence_metadata,
    normalize_retrieved_context,
)
from src.ingest.clinicaltrials import search_trials
from src.kb.chunking import docs_to_chunks, validate_chunk_traceability
from src.models import EvidenceDoc
from src.retrieval.hybrid import RECORD_TYPE_WEIGHT, _is_citable_for_level_boost
from src.tracks.pipeline import _qualifies_for_expected_levels


class EvidenceMetadataTests(unittest.TestCase):
    def test_classify_trial_registry_not_rct(self) -> None:
        level, eligible = classify_trial_registry("RECRUITING")
        self.assertEqual(level, "other")
        self.assertFalse(eligible)

        level_done, eligible_done = classify_trial_registry("COMPLETED")
        self.assertEqual(level_done, "other")
        self.assertTrue(eligible_done)

    def test_normalize_legacy_clinicaltrials_json(self) -> None:
        doc = EvidenceDoc(
            doc_id="nct:NCT00000001",
            source="clinicaltrials",
            title="Example trial",
            text="A recruiting hypertension trial.",
            evidence_level="rct",
            extra={"status": "RECRUITING"},
        )
        fixed = normalize_evidence_metadata(doc)
        self.assertEqual(fixed.record_type, "trial_registry")
        self.assertEqual(fixed.evidence_level, "other")
        self.assertFalse(fixed.citation_eligible)

    def test_chunks_carry_metadata_and_extra(self) -> None:
        doc = normalize_evidence_metadata(
            EvidenceDoc(
                doc_id="nct:NCT00000002",
                source="clinicaltrials",
                title="Completed trial",
                text="Trial summary text.",
                url="https://clinicaltrials.gov/study/NCT00000002",
                extra={"status": "COMPLETED", "primary_outcome": "BP change"},
            )
        )
        chunks = docs_to_chunks([doc])
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.record_type, "trial_registry")
        self.assertTrue(chunk.citation_eligible)
        self.assertIn("clinicaltrials.gov", chunk.source_locator)
        self.assertEqual(chunk.extra.get("status"), "COMPLETED")
        issues = validate_chunk_traceability(chunks)
        self.assertTrue(issues["ok"])

    def test_trial_registry_does_not_satisfy_clinical_expect_levels(self) -> None:
        ctx = normalize_retrieved_context(
            {
                "doc_id": "nct:NCT00000003",
                "source": "clinicaltrials",
                "evidence_level": "rct",
                "record_type": "trial_registry",
                "citation_eligible": False,
                "trial_status": "RECRUITING",
            }
        )
        self.assertEqual(ctx["evidence_level"], "other")
        self.assertFalse(
            _qualifies_for_expected_levels(ctx, ("guideline", "meta", "rct"))
        )

    def test_retrieved_context_fallback_without_record_type(self) -> None:
        ctx = normalize_retrieved_context(
            {
                "chunk_id": "nct:NCT00000004#c0",
                "doc_id": "nct:NCT00000004",
                "source": "clinicaltrials",
                "evidence_level": "rct",
                "trial_status": "NOT_YET_RECRUITING",
            }
        )
        self.assertEqual(ctx["record_type"], "trial_registry")
        self.assertEqual(ctx["evidence_level"], "other")
        self.assertFalse(ctx["citation_eligible"])

    def test_trial_registry_penalized_in_hybrid_scoring(self) -> None:
        self.assertLess(RECORD_TYPE_WEIGHT["trial_registry"], 1.0)
        self.assertFalse(
            _is_citable_for_level_boost(
                {"record_type": "trial_registry", "citation_eligible": True, "evidence_level": "other"}
            )
        )

    def test_clinicaltrials_ingest_sets_registry_fields(self) -> None:
        with unittest.mock.patch("src.ingest.clinicaltrials.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value.json.return_value = {
                "studies": [
                    {
                        "protocolSection": {
                            "identificationModule": {
                                "nctId": "NCT99999999",
                                "briefTitle": "Hypertension diet trial",
                            },
                            "descriptionModule": {"briefSummary": "Testing sodium reduction."},
                            "statusModule": {
                                "overallStatus": "RECRUITING",
                                "startDateStruct": {"date": "2024-01"},
                            },
                            "conditionsModule": {"conditions": ["Hypertension"]},
                            "armsInterventionsModule": {"interventions": [{"name": "DASH diet"}]},
                        }
                    }
                ]
            }
            mock_client.return_value.__enter__.return_value.get.return_value.raise_for_status = lambda: None
            docs = search_trials("Hypertension", page_size=1)

        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc.record_type, "trial_registry")
        self.assertEqual(doc.evidence_level, "other")
        self.assertFalse(doc.citation_eligible)
        self.assertIn("clinicaltrials.gov", doc.source_locator)


if __name__ == "__main__":
    unittest.main()
