"""证据优先级统一口径的回归测试。"""

from __future__ import annotations

import unittest

from src.kb.weights import combined_priority, evidence_priority
from src.retrieval.hybrid import score_evidence_priority


class EvidencePriorityTests(unittest.TestCase):
    def test_retrieval_uses_shared_priority_formula(self) -> None:
        item = {"evidence_level": "rct", "year": "2024"}

        self.assertEqual(
            score_evidence_priority(item),
            combined_priority("rct", 2024),
        )

    def test_priority_normalizes_level_and_invalid_year(self) -> None:
        self.assertEqual(evidence_priority(" GUIDELINE "), evidence_priority("guideline"))
        self.assertEqual(
            score_evidence_priority({"evidence_level": "meta", "year": "unknown"}),
            combined_priority("meta", None),
        )


if __name__ == "__main__":
    unittest.main()
