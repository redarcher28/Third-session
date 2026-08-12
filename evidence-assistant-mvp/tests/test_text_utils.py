# -*- coding: utf-8 -*-
"""文本展示截断工具测试。"""

from __future__ import annotations

import unittest

from src.generation.answer import contexts_to_citations, format_reference_section
from src.models import Citation
from src.text_utils import citation_display_fields, context_to_citation_kwargs, truncate_at_sentence


class TextUtilsTests(unittest.TestCase):
    def test_truncate_at_chinese_sentence_boundary(self) -> None:
        text = "第一句完整内容。第二句也很长，但不应被截断到半句。"
        out = truncate_at_sentence(text, 18)
        self.assertTrue(out.endswith("。"))
        self.assertNotIn("第二句", out)

    def test_truncate_at_english_sentence_boundary(self) -> None:
        text = "First sentence is complete. Second sentence should not appear."
        out = truncate_at_sentence(text, 35)
        self.assertTrue(out.endswith(".") or out.endswith("…"))
        self.assertNotIn("Second", out.replace("…", ""))

    def test_citation_display_fields_keep_full_text(self) -> None:
        long = "A" * 500 + "。结尾句。"
        full, snippet = citation_display_fields(long, snippet_max=120)
        self.assertEqual(full, long)
        self.assertLessEqual(len(snippet), 123)
        self.assertTrue(snippet.endswith("。") or snippet.endswith("…"))

    def test_context_to_citation_includes_text_field(self) -> None:
        ctx = {
            "doc_id": "local:test",
            "title": "测试",
            "source": "local",
            "text": "指南建议长期管理。具体目标需个体化评估。",
            "evidence_level": "guideline",
        }
        fields = context_to_citation_kwargs(ctx, 1)
        self.assertEqual(fields["text"], ctx["text"])
        self.assertIn("。", fields["snippet"])

    def test_contexts_to_citations_populates_text_for_ui(self) -> None:
        cites = contexts_to_citations(
            [
                {
                    "doc_id": "pmid:1",
                    "title": "Study",
                    "source": "pubmed",
                    "text": "Background info. Methods were described in detail.",
                    "evidence_level": "rct",
                }
            ]
        )
        self.assertEqual(cites[0].text, "Background info. Methods were described in detail.")
        self.assertTrue(cites[0].snippet.endswith(".") or cites[0].snippet.endswith("…"))

    def test_reference_section_uses_sentence_aligned_preview(self) -> None:
        long_tail = "。".join([f"补充句{i}。" for i in range(1, 80)])
        full = f"第一句完整。第二句完整。{long_tail}"
        cite = Citation(
            index=1,
            doc_id="local:x",
            title="指南",
            source="local",
            text=full,
            snippet=full,
        )
        section = format_reference_section([cite])
        preview_line = section.split("摘要：")[1].split("\n")[0]
        self.assertLess(len(preview_line), len(full))
        self.assertTrue(preview_line.endswith("。") or preview_line.endswith("…"))


if __name__ == "__main__":
    unittest.main()
