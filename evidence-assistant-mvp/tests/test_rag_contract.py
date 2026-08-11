# -*- coding: utf-8 -*-
"""B 组 RAG 数据流契约测试。

这些测试使用内存假检索器/假 LLM，只验证“问题 → 检索 → 证据 Prompt → 引用 →
Open WebUI 回显”的连接关系，不写入 A 组知识库或 Chroma 数据。
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.app.openwebui import (
    OpenAIChatRequest,
    OpenAIMessage,
    _stream_rag_answer,
    chat_completions,
)
from src.generation.answer import generate_answer
from src.models import AskResponse, Citation
from src.retrieval.hybrid import _tokenize
from src.tracks.pipeline import ask
from src.tracks.prompt_profiles import build_synthesis_messages
from src.tracks.nutrition import append_nutrition_query_aliases, detect_dosage_request


def _citation(index: int = 1) -> Citation:
    return Citation(
        index=index,
        doc_id="local:hypertension-guideline",
        title="高血压管理指南摘要",
        source="local",
        year=2024,
        url="https://example.test/hypertension-guideline",
        evidence_level="guideline",
        snippet="指南指出，长期管理需要结合血压控制目标与整体心血管风险。",
    )


def _context() -> dict[str, object]:
    return {
        "chunk_id": "local:hypertension-guideline#c0",
        "doc_id": "local:hypertension-guideline",
        "source": "local",
        "title": "高血压管理指南摘要",
        "text": "指南指出，长期管理需要结合血压控制目标与整体心血管风险。",
        "year": 2024,
        "url": "https://example.test/hypertension-guideline",
        "evidence_level": "guideline",
        "tags": "hypertension",
    }


class _FakeLLM:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def chat(self, messages: list[dict[str, str]], **_: object) -> str:
        self.messages = messages
        return "证据支持长期管理有助于控制相关风险[1]。"


class _FakeRetriever:
    def __init__(self, contexts: list[dict[str, object]]) -> None:
        self.contexts = contexts
        self.query = ""
        self.kwargs: dict[str, object] = {}

    def retrieve(self, query: str, **kwargs: object) -> list[dict[str, object]]:
        self.query = query
        self.kwargs = kwargs
        return list(self.contexts)


class RagContractTests(unittest.TestCase):
    def test_synthesis_prompt_contains_retrieved_evidence_and_citation_contract(self) -> None:
        messages = build_synthesis_messages(
            "clinical",
            "高血压为什么有时需要长期管理？",
            "[1] (guideline) 高血压管理指南摘要 | local | 2024 | local:hypertension-guideline\n"
            "URL: https://example.test/hypertension-guideline\n"
            "指南指出，长期管理需要结合血压控制目标与整体心血管风险。",
        )
        combined = "\n".join(message["content"] for message in messages)

        self.assertIn("只能依据下方 RETRIEVED EVIDENCE", combined)
        self.assertIn("高血压管理指南摘要", combined)
        self.assertIn("[1]", combined)
        self.assertIn("不要为了完整而猜测", combined)

    def test_generation_passes_retrieved_context_to_llm_and_returns_citations(self) -> None:
        fake_llm = _FakeLLM()
        with patch("src.generation.answer.get_llm", return_value=fake_llm):
            answer, citations, refused = generate_answer(
                "高血压为什么有时需要长期管理？",
                [_context()],
                system_persona="临床证据助手",
                answer_style="结论与证据概览",
                track="clinical",
            )

        user_prompt = fake_llm.messages[1]["content"]
        self.assertIn("高血压管理指南摘要", user_prompt)
        self.assertIn("长期管理需要结合血压控制目标", user_prompt)
        self.assertIn("[1]", answer)
        self.assertEqual(citations[0].doc_id, "local:hypertension-guideline")
        self.assertFalse(refused)

    def test_pipeline_connects_rewrite_retrieval_generation_and_validation(self) -> None:
        retriever = _FakeRetriever([_context()])
        citation = _citation()
        with (
            patch(
                "src.tracks.pipeline.rewrite_clinical_query",
                return_value="hypertension guideline long-term management cardiovascular risk",
            ),
            patch(
                "src.tracks.pipeline.generate_answer",
                return_value=("证据支持长期管理有助于控制相关风险[1]。", [citation], False),
            ),
            patch(
                "src.tracks.pipeline.get_settings",
                return_value=SimpleNamespace(
                    rag_use_llm_query_rewrite=True,
                    rag_use_llm_rerank=False,
                ),
            ),
        ):
            response = ask(
                "高血压为什么有时需要长期管理？",
                track="clinical",
                top_k=3,
                retriever=retriever,
            )

        self.assertEqual(retriever.query, "hypertension guideline long-term management cardiovascular risk")
        self.assertEqual(retriever.kwargs["top_k"], 3)
        self.assertFalse(retriever.kwargs["use_llm_rerank"])
        self.assertEqual(response.retrieval["retrieved_count"], 1)
        self.assertEqual(response.citations[0].index, 1)
        self.assertEqual(response.contexts[0].doc_id, "local:hypertension-guideline")
        self.assertTrue(response.citation_check["ok"])
        self.assertFalse(response.refused)
        self.assertIn("total_ms", response.timings_ms)

    def test_openwebui_response_keeps_answer_to_source_mapping_visible(self) -> None:
        citation = _citation()
        result = AskResponse(
            answer="证据支持长期管理有助于控制相关风险[1]。",
            citations=[citation],
            contexts=[citation],
            track="clinical",
            prompt_version="test-version",
            retrieval={
                "retrieved_count": 1,
                "rewritten_query": "hypertension guideline long-term management",
                "sources": {"local": 1},
                "evidence_levels": {"guideline": 1},
            },
            citation_check={"ok": True, "used_brackets": [1]},
        )
        request = OpenAIChatRequest(
            model="evidence-clinical",
            messages=[
                OpenAIMessage(role="system", content="Open WebUI context"),
                OpenAIMessage(
                    role="user",
                    content=[{"type": "text", "text": "高血压为什么有时需要长期管理？"}],
                ),
            ],
        )

        with patch("src.app.openwebui.ask", return_value=result) as mocked_ask:
            payload = chat_completions(request)

        self.assertIsInstance(payload, dict)
        content = payload["choices"][0]["message"]["content"]  # type: ignore[index]
        self.assertIn("[1]", content)
        self.assertIn("### 证据面板", content)
        self.assertIn("### 证据来源", content)
        self.assertIn("#### [1] 高血压管理指南摘要", content)
        self.assertIn("高血压管理指南摘要", content)
        self.assertIn("摘要：指南指出，长期管理需要结合血压控制目标与整体心血管风险。", content)
        self.assertIn("改写查询：`hypertension guideline long-term management`", content)
        self.assertIn("来源分布：local 1 条", content)
        self.assertIn("### 引用校验", content)
        self.assertIn("https://example.test/hypertension-guideline", content)
        mocked_ask.assert_called_once_with(
            "高血压为什么有时需要长期管理？",
            track="clinical",
            top_k=5,
            use_live_tools=False,
        )

    def test_openwebui_stream_emits_native_sources_without_repeating_footer(self) -> None:
        citation = _citation()
        result = AskResponse(
            answer="证据支持长期管理有助于控制相关风险[1]。",
            citations=[citation],
            contexts=[citation],
            track="clinical",
            prompt_version="test-version",
            retrieval={"retrieved_count": 1},
            citation_check={"ok": True, "used_brackets": [1]},
        )

        with patch("src.app.openwebui.ask", return_value=result) as mocked_ask:
            frames = list(
                _stream_rag_answer(
                    response_id="chatcmpl-test",
                    created=1,
                    model="evidence-clinical",
                    question="高血压为什么有时需要长期管理？",
                    track="clinical",
                )
            )

        payloads = []
        for frame in frames:
            if not frame.startswith("data: "):
                continue
            raw = frame[len("data: ") :].strip()
            if raw != "[DONE]":
                payloads.append(json.loads(raw))

        source_payload = next(payload for payload in payloads if "event" in payload)
        source = source_payload["event"]
        self.assertEqual(source["type"], "source")
        self.assertEqual(len(source["data"]["document"]), 1)
        self.assertEqual(len(source["data"]["metadata"]), 1)
        self.assertEqual(
            source["data"]["metadata"][0]["url"],
            "https://example.test/hypertension-guideline",
        )
        answer_text = "".join(
            payload.get("choices", [{}])[0].get("delta", {}).get("content", "")
            for payload in payloads
            if payload.get("object") == "chat.completion.chunk"
        )
        self.assertIn("[1]", answer_text)
        self.assertNotIn("### 证据来源", answer_text)
        mocked_ask.assert_called_once_with(
            "高血压为什么有时需要长期管理？",
            track="clinical",
            top_k=5,
            use_live_tools=False,
        )

    def test_refusal_keeps_retrieved_contexts_for_evidence_panel(self) -> None:
        context = _context()
        context["evidence_level"] = "other"
        retriever = _FakeRetriever([context])
        with patch(
            "src.tracks.pipeline.rewrite_clinical_query",
            return_value="hypertension lifestyle evidence",
        ), patch(
            "src.tracks.pipeline.get_settings",
            return_value=SimpleNamespace(
                rag_use_llm_query_rewrite=True,
                rag_use_llm_rerank=False,
            ),
        ):
            response = ask(
                "高血压应该注意什么？",
                track="clinical",
                retriever=retriever,
            )

        self.assertTrue(response.refused)
        self.assertEqual(response.citation_check["reason"], "missing_evidence_type")
        self.assertEqual(len(response.contexts), 1)
        self.assertEqual(response.contexts[0].title, "高血压管理指南摘要")

    def test_latest_main_nutrition_aliases_reach_chinese_bm25_tokens(self) -> None:
        rewritten = append_nutrition_query_aliases("膳食纤维对心血管风险有什么研究提示？", "膳食纤维")
        tokens = _tokenize("膳食纤维与全谷物")

        self.assertIn("dietary fiber whole grains cardiovascular risk", rewritten)
        self.assertIn("膳食", tokens)
        self.assertIn("食纤", tokens)

    def test_nutrition_dosage_request_is_blocked_before_llm(self) -> None:
        question = "我每天应该服用多少毫克降压药？"
        self.assertTrue(detect_dosage_request(question))

        retriever = _FakeRetriever([])
        with patch("src.tracks.pipeline.rewrite_nutrition_query") as rewrite:
            response = ask(question, track="nutrition", retriever=retriever)

        self.assertTrue(response.refused)
        self.assertEqual(response.citation_check["reason"], "dosage_request")
        self.assertEqual(retriever.query, "")
        rewrite.assert_not_called()

    def test_rebuilt_latest_main_corpus_is_available_to_rag(self) -> None:
        from src.kb.store import EvidenceStore

        store = EvidenceStore()

        self.assertGreaterEqual(store.count(), 11000)


if __name__ == "__main__":
    unittest.main()
