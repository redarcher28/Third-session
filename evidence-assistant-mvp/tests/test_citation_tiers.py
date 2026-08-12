# -*- coding: utf-8 -*-
"""citation_eligible 分级与合成过滤测试。"""

from __future__ import annotations

import unittest

from src.generation.answer import compute_faithfulness_proxy
from src.ingest import apply_citation_eligibility, classify_trial_registry, resolve_context_citation_eligible
from src.models import EvidenceDoc
from src.text_utils import filter_citable_contexts


class CitationTierTests(unittest.TestCase):
    def test_local_doc_without_url_not_citable(self) -> None:
        doc = EvidenceDoc(
            doc_id="local:seed",
            title="本地种子",
            source="local",
            text="some local text",
            url="",
            doi="",
        )
        doc = apply_citation_eligibility(doc)
        self.assertFalse(doc.citation_eligible)

    def test_pubmed_with_doi_is_citable(self) -> None:
        doc = EvidenceDoc(
            doc_id="pmid:123",
            title="Trial",
            source="pubmed",
            text="abstract",
            url="",
            doi="10.1000/example",
        )
        doc = apply_citation_eligibility(doc)
        self.assertTrue(doc.citation_eligible)

    def test_trial_registry_completed_vs_recruiting(self) -> None:
        level_done, eligible_done = classify_trial_registry("COMPLETED")
        level_rec, eligible_rec = classify_trial_registry("RECRUITING")
        self.assertEqual(level_done, "other")
        self.assertTrue(eligible_done)
        self.assertFalse(eligible_rec)

    def test_filter_citable_contexts(self) -> None:
        contexts = [
            {"doc_id": "a", "citation_eligible": True, "text": "yes"},
            {"doc_id": "b", "citation_eligible": False, "text": "no"},
        ]
        filtered = filter_citable_contexts(contexts)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["doc_id"], "a")

    def test_resolve_context_citation_eligible_for_stale_local(self) -> None:
        item = {
            "doc_id": "local:seed",
            "source": "local",
            "citation_eligible": True,
            "url": "",
        }
        self.assertFalse(resolve_context_citation_eligible(item))

    def test_faithfulness_proxy_positive_overlap(self) -> None:
        contexts = [{"text": "DASH diet sodium reduction blood pressure guideline"}]
        answer = "DASH 饮食限钠有助于降低血压 [1]。"
        score = compute_faithfulness_proxy(answer, contexts)
        self.assertGreater(score, 0.1)


if __name__ == "__main__":
    unittest.main()
