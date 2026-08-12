# -*- coding: utf-8 -*-
"""知识库健康与引用正文校验测试。"""

from __future__ import annotations

import unittest

from src.generation.answer import append_reference_section_if_needed, finalize_grounded_answer
from src.kb.health import kb_health_report, probe_retrieval_index, probe_sqlite_integrity
from src.models import Citation
from src.tools.cite_check import answer_body_for_citation_check, verify_citations


class KbHealthTests(unittest.TestCase):
    def test_health_report_has_status_fields(self) -> None:
        report = kb_health_report()
        self.assertIn(report["status"], {"ok", "degraded"})
        self.assertIn("chroma", report)
        self.assertIn("retrieval", report)
        self.assertIn("sqlite", report)
        self.assertIn("warnings", report)

    def test_sqlite_probe_has_structured_result(self) -> None:
        probe = probe_sqlite_integrity()
        self.assertIn("ok", probe)
        self.assertIn("warning", probe)
        self.assertIn("error", probe)
    def test_retrieval_index_uses_full_bm25_limit(self) -> None:
        probe = probe_retrieval_index()
        self.assertIn("bm25_indexed_count", probe)
        self.assertIn("bm25_cache_total", probe)
        if probe["bm25_cache_total"] > 5000:
            self.assertGreaterEqual(probe["bm25_indexed_count"], 5000)


class CitationBodyVerifyTests(unittest.TestCase):
    def test_auto_reference_section_does_not_count_as_body_citation(self) -> None:
        contexts = [{"doc_id": "local:a", "title": "A", "text": "evidence"}]
        body = "这是一个没有任何引用的结论。"
        answer = append_reference_section_if_needed(
            body,
            [
                Citation(
                    index=1,
                    doc_id="local:a",
                    title="A",
                    source="local",
                    text="evidence",
                    snippet="evidence",
                )
            ],
            used_indices=[1],
        )
        check = verify_citations(answer, contexts)
        self.assertFalse(check["body_has_citations"])
        self.assertFalse(check["ok"])

    def test_finalize_only_appends_used_references(self) -> None:
        citations = [
            Citation(index=1, doc_id="local:a", title="A", source="local", text="a", snippet="a"),
            Citation(index=2, doc_id="local:b", title="B", source="local", text="b", snippet="b"),
        ]
        answer = "结论 supported [1]。"
        check = verify_citations(answer, [{"doc_id": "local:a"}, {"doc_id": "local:b"}])
        final, used, check = finalize_grounded_answer(answer, citations, check)
        self.assertIn("参考文献", final)
        self.assertIn("[1]", final)
        self.assertNotIn("[2]", final.split("参考文献")[1])
        self.assertEqual(len(used), 1)
        self.assertEqual(used[0].index, 1)

    def test_answer_body_strips_disclaimer(self) -> None:
        from src.generation.answer import DISCLAIMER

        text = f"正文 [1]。{DISCLAIMER}"
        body = answer_body_for_citation_check(text)
        self.assertIn("[1]", body)
        self.assertNotIn("声明", body)


if __name__ == "__main__":
    unittest.main()
