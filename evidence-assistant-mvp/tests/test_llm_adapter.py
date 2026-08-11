# -*- coding: utf-8 -*-
"""LLM 协议适配单测，不访问真实供应商。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from src.llm import LLMClient


def _settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "llm_api_format": "anthropic",
        "llm_api_key": "agentrouter-test-token",
        "llm_base_url": "https://co.agentrouter.org",
        "llm_model": "claude-opus-5",
        "embedding_mode": "auto",
        "embedding_model": "text-embedding-3-small",
        "embedding_api_key": "",
        "embedding_base_url": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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
