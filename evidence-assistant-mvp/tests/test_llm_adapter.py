# -*- coding: utf-8 -*-
"""LLM 协议适配单测，不访问真实供应商。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx

from src.llm import LLMClient


def _settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "llm_api_format": "anthropic",
        "llm_api_key": "agentrouter-test-token",
        "llm_base_url": "https://co.agentrouter.org",
        "llm_model": "claude-opus-5",
        "llm_reasoning_effort": "",
        "embedding_mode": "auto",
        "embedding_model": "text-embedding-3-small",
        "embedding_api_key": "",
        "embedding_base_url": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __enter__(self) -> "_FakeStreamResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        yield from self._lines


class LLMAdapterTests(unittest.TestCase):
    def test_anthropic_messages_request_and_local_embedding_fallback(self) -> None:
        response = httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "OK"}]},
            request=httpx.Request("POST", "https://co.agentrouter.org/v1/messages"),
        )
        with (
            patch("src.llm.get_settings", return_value=_settings()),
            patch("src.llm.httpx.post", return_value=response) as post,
        ):
            client = LLMClient()
            answer = client.chat(
                [
                    {"role": "system", "content": "只回答证据问题"},
                    {"role": "user", "content": "高血压与限盐？"},
                ],
                max_tokens=80,
            )
            vectors = client.embed(["限盐"])

        self.assertEqual(answer, "OK")
        self.assertFalse(client.is_offline)
        self.assertFalse(client.has_remote_embeddings)
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 384)
        post.assert_called_once()
        url = post.call_args.args[0]
        kwargs = post.call_args.kwargs
        self.assertEqual(url, "https://co.agentrouter.org/v1/messages")
        self.assertEqual(kwargs["json"]["model"], "claude-opus-5")
        self.assertEqual(kwargs["json"]["system"], "只回答证据问题")
        self.assertEqual(
            kwargs["json"]["messages"],
            [{"role": "user", "content": "高血压与限盐？"}],
        )
        self.assertEqual(kwargs["headers"]["x-api-key"], "agentrouter-test-token")

    def test_responses_request_uses_byeapi_root_and_reasoning_effort(self) -> None:
        response = httpx.Response(
            200,
            json={"output_text": "OK"},
            request=httpx.Request("POST", "https://api.byeapi.top/v1/responses"),
        )
        with (
            patch(
                "src.llm.get_settings",
                return_value=_settings(
                    llm_api_format="responses",
                    llm_api_key="byeapi-test-token",
                    llm_base_url="https://api.byeapi.top",
                    llm_model="gpt-5.6-luna",
                    llm_reasoning_effort="xhigh",
                ),
            ),
            patch("src.llm.httpx.post", return_value=response) as post,
        ):
            client = LLMClient()
            answer = client.chat(
                [
                    {"role": "system", "content": "只回答证据问题"},
                    {"role": "user", "content": "限盐与血压？"},
                ],
                max_tokens=80,
            )
            vectors = client.embed(["限盐"])

        self.assertEqual(answer, "OK")
        self.assertEqual(client.api_format, "responses")
        self.assertFalse(client.has_remote_embeddings)
        self.assertEqual(len(vectors[0]), 384)
        post.assert_called_once()
        url = post.call_args.args[0]
        kwargs = post.call_args.kwargs
        self.assertEqual(url, "https://api.byeapi.top/v1/responses")
        self.assertEqual(kwargs["json"]["model"], "gpt-5.6-luna")
        self.assertEqual(kwargs["json"]["max_output_tokens"], 80)
        self.assertEqual(kwargs["json"]["reasoning"], {"effort": "xhigh"})
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer byeapi-test-token")

    def test_responses_stream_yields_text_deltas(self) -> None:
        response = _FakeStreamResponse(
            [
                'event: response.output_text.delta',
                'data: {"type":"response.output_text.delta","delta":"第一段"}',
                "",
                'data: {"type":"response.output_text.delta","delta":"第二段"}',
                "",
                "data: [DONE]",
                "",
            ]
        )
        stream_context = MagicMock()
        stream_context.__enter__.return_value = response
        stream_context.__exit__.return_value = None
        with (
            patch(
                "src.llm.get_settings",
                return_value=_settings(
                    llm_api_format="responses",
                    llm_api_key="byeapi-test-token",
                    llm_base_url="https://api.byeapi.top",
                    llm_model="gpt-5.6-luna",
                ),
            ),
            patch("src.llm.httpx.stream", return_value=stream_context) as stream,
        ):
            client = LLMClient()
            chunks = list(
                client.stream_chat(
                    [{"role": "user", "content": "限盐与血压？"}],
                    max_tokens=80,
                )
            )

        self.assertEqual(chunks, ["第一段", "第二段"])
        kwargs = stream.call_args.kwargs
        self.assertTrue(kwargs["json"]["stream"])
        self.assertEqual(kwargs["headers"]["accept"], "text/event-stream")

    def test_openai_mode_keeps_remote_embedding_default(self) -> None:
        with patch(
            "src.llm.get_settings",
            return_value=_settings(
                llm_api_format="openai",
                llm_base_url="https://api.openai.com/v1",
                llm_model="gpt-4o-mini",
            ),
        ):
            client = LLMClient()

        self.assertEqual(client.api_format, "openai")
        self.assertTrue(client.has_remote_embeddings)

    def test_deepseek_openai_mode_uses_local_embedding_by_default(self) -> None:
        with patch(
            "src.llm.get_settings",
            return_value=_settings(
                llm_api_format="openai",
                llm_api_key="deepseek-test-token",
                llm_base_url="https://api.deepseek.com",
                llm_model="deepseek-v4-flash",
            ),
        ):
            client = LLMClient()

        self.assertEqual(client.api_format, "openai")
        self.assertFalse(client.has_remote_embeddings)
        self.assertEqual(client.embedding_mode, "local")

    def test_placeholder_token_stays_offline(self) -> None:
        with patch(
            "src.llm.get_settings",
            return_value=_settings(llm_api_key="fill-your-agentrouter-token-here"),
        ):
            client = LLMClient()

        self.assertTrue(client.is_offline)
        self.assertFalse(client.has_remote_embeddings)


if __name__ == "__main__":
    unittest.main()
