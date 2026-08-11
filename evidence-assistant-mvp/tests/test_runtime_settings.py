# -*- coding: utf-8 -*-
"""本机模型连接设置的存储、脱敏和热更新契约测试。"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from src.app.settings_api import ModelConnectionUpdate, update_settings
from src.runtime_config import (
    RuntimeLLMConfig,
    clear_runtime_config,
    load_runtime_config,
    save_runtime_config,
)


def _request(host: str = "127.0.0.1") -> SimpleNamespace:
    return SimpleNamespace(client=SimpleNamespace(host=host))


def _settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "llm_api_format": "responses",
        "llm_api_key": "old-token",
        "llm_base_url": "https://old.example",
        "llm_model": "old-model",
        "llm_reasoning_effort": "",
        "openwebui_url": "http://127.0.0.1:8080/",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RuntimeSettingsTests(unittest.TestCase):
    def test_runtime_file_is_private_and_round_trips_without_printing_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "llm_runtime.json"
            config = RuntimeLLMConfig(
                api_format="responses",
                api_key="secret-token-for-test",
                base_url="https://api.example/v1",
                model="gpt-test",
                reasoning_effort="xhigh",
            )
            with patch.dict(os.environ, {"EVIDENCE_RUNTIME_CONFIG": str(path)}):
                save_runtime_config(config)
                loaded = load_runtime_config()
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded.model, "gpt-test")  # type: ignore[union-attr]
                self.assertEqual(loaded.api_key, "secret-token-for-test")  # type: ignore[union-attr]
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                self.assertTrue(clear_runtime_config())
                self.assertIsNone(load_runtime_config())

    def test_update_saves_key_and_url_then_refreshes_client_without_returning_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "llm_runtime.json"
            initial = _settings()
            updated = _settings(
                llm_api_key="new-token-1234",
                llm_base_url="https://api.byeapi.top",
                llm_model="gpt-5.6-luna",
                llm_reasoning_effort="xhigh",
            )
            with (
                patch.dict(os.environ, {"EVIDENCE_RUNTIME_CONFIG": str(path)}),
                patch("src.app.settings_api.get_settings", side_effect=[initial, updated]),
                patch("src.app.settings_api.reset_llm") as reset,
            ):
                result = update_settings(
                    _request(),
                    ModelConnectionUpdate(
                        api_format="responses",
                        base_url="https://api.byeapi.top/",
                        model="gpt-5.6-luna",
                        reasoning_effort="xhigh",
                        api_key="new-token-1234",
                    ),
                )

            with patch.dict(os.environ, {"EVIDENCE_RUNTIME_CONFIG": str(path)}):
                loaded = load_runtime_config()
            self.assertEqual(loaded.base_url, "https://api.byeapi.top")  # type: ignore[union-attr]
            self.assertEqual(loaded.model, "gpt-5.6-luna")  # type: ignore[union-attr]
            self.assertTrue(result["status"]["api_key_configured"])
            self.assertEqual(result["status"]["api_key_hint"], "••••••••1234")
            self.assertNotIn("api_key", result["status"])
            self.assertNotIn("reasoning_effort", result["status"])
            self.assertEqual(loaded.reasoning_effort, "")  # type: ignore[union-attr]
            reset.assert_called_once_with()

    def test_remote_request_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as context:
            update_settings(_request("192.168.1.20"), ModelConnectionUpdate())
        self.assertEqual(context.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
