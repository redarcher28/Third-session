# -*- coding: utf-8 -*-
"""Open WebUI 证据台品牌配置的纯本地回归测试。"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.configure_openwebui import (
    EVIDENCE_BANNER_ID,
    EVIDENCE_SUGGESTIONS,
    apply_evidence_ui_config,
)


class OpenWebUIConfigTests(unittest.TestCase):
    def test_merge_preserves_existing_ui_content_and_adds_evidence_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "webui.db"
            connection = sqlite3.connect(db_path)
            connection.execute(
                "CREATE TABLE config (key TEXT PRIMARY KEY, value JSON NOT NULL, updated_at INTEGER)"
            )
            connection.execute(
                "INSERT INTO config (key, value, updated_at) VALUES (?, ?, ?)",
                (
                    "ui.banners",
                    json.dumps(
                        [
                            {
                                "id": "existing-banner",
                                "type": "warning",
                                "content": "用户自己的提醒",
                            }
                        ],
                        ensure_ascii=False,
                    ),
                    1,
                ),
            )
            connection.execute(
                "INSERT INTO config (key, value, updated_at) VALUES (?, ?, ?)",
                (
                    "ui.prompt_suggestions",
                    json.dumps(
                        [{"title": ["已有建议"], "content": "保留这条建议"}],
                        ensure_ascii=False,
                    ),
                    1,
                ),
            )
            connection.commit()
            connection.close()

            result = apply_evidence_ui_config(db_path, timestamp=123)
            self.assertTrue(result["updated"])

            connection = sqlite3.connect(db_path)
            rows = dict(connection.execute("SELECT key, value FROM config").fetchall())
            connection.close()

            banners = json.loads(rows["ui.banners"])
            suggestions = json.loads(rows["ui.prompt_suggestions"])
            self.assertEqual(banners[0]["id"], EVIDENCE_BANNER_ID)
            self.assertEqual(banners[0]["timestamp"], 123)
            self.assertIn("existing-banner", {item["id"] for item in banners[1:]})
            self.assertEqual(len(suggestions), len(EVIDENCE_SUGGESTIONS) + 1)
            self.assertEqual(suggestions[-1]["content"], "保留这条建议")
            self.assertEqual(json.loads(rows["evaluation.arena.enable"]), False)


if __name__ == "__main__":
    unittest.main()
