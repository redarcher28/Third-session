# -*- coding: utf-8 -*-
"""ReAct 智能体与 pipeline 查询改写对齐测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.agent.react_agent import ReactEvidenceAgent, react_chat
from src.retrieval.hybrid import HybridRetriever
from src.tracks.pipeline import reformulate_query
from src.tracks.prompt_profiles import PROMPT_VERSION


class ReactQueryRewriteTests(unittest.TestCase):
    def test_clinical_lexical_rewrite_enables_bm25_hits(self) -> None:
        question = "DASH饮食对血压的证据？"
        rewritten, mode = reformulate_query(question, "clinical")
        self.assertEqual(mode, "lexical")
        self.assertIn("guideline", rewritten)
        hits = HybridRetriever().retrieve(rewritten, top_k=3)
        self.assertGreater(len(hits), 0)

    def test_react_agent_uses_pipeline_rewrite_and_returns_retrieval_meta(self) -> None:
        agent = ReactEvidenceAgent(track="clinical", top_k=3, use_live_tools=False)
        # 离线/格式异常路径：无 Action 时会用改写查询做默认检索
        llm_patches = patch(
            "src.agent.react_agent.get_llm",
            return_value=type(
                "OfflineLLM",
                (),
                {"chat": lambda self, *a, **k: "无法解析的回复"},
            )(),
        )
        gen_patch = patch(
            "src.agent.react_agent.generate_answer",
            return_value=("测试回答", [], False),
        )
        cite_patch = patch(
            "src.agent.react_agent.verify_citations",
            return_value={"ok": True, "has_citations": True},
        )
        strip_patch = patch(
            "src.agent.react_agent.strip_invalid_claims",
            side_effect=lambda answer, _check: answer,
        )
        with llm_patches, gen_patch, cite_patch, strip_patch:
            result = agent.run("DASH饮食对血压的证据？")
        self.assertIn("guideline", result.get("rewritten_query", ""))
        self.assertGreater(len(result.get("contexts", [])), 0)
        self.assertEqual(result.get("prompt_version"), PROMPT_VERSION)
        retrieval = result.get("retrieval") or {}
        self.assertGreater(retrieval.get("retrieved_count", 0), 0)

    def test_react_chat_empty_messages_includes_prompt_version(self) -> None:
        payload = react_chat([])
        self.assertEqual(payload.get("prompt_version"), PROMPT_VERSION)


if __name__ == "__main__":
    unittest.main()
