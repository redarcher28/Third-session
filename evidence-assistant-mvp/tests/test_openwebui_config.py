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

            result = apply_evidence_ui_config(
                db_path,
                timestamp=123,
                settings_url="http://127.0.0.1:8000/settings",
            )
            self.assertTrue(result["updated"])

            connection = sqlite3.connect(db_path)
            rows = dict(connection.execute("SELECT key, value FROM config").fetchall())
            connection.close()

            banners = json.loads(rows["ui.banners"])
            suggestions = json.loads(rows["ui.prompt_suggestions"])
            self.assertEqual(banners[0]["id"], EVIDENCE_BANNER_ID)
            self.assertEqual(banners[0]["timestamp"], 123)
            self.assertIn("http://127.0.0.1:8000/settings", banners[0]["content"])
            self.assertIn("existing-banner", {item["id"] for item in banners[1:]})
            self.assertEqual(len(suggestions), len(EVIDENCE_SUGGESTIONS) + 1)
            self.assertEqual(suggestions[-1]["content"], "保留这条建议")
            self.assertEqual(json.loads(rows["evaluation.arena.enable"]), False)

    def test_merge_works_with_openwebui_06_blob_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "webui.db"
            connection = sqlite3.connect(db_path)
            connection.execute(
                "CREATE TABLE config (id INTEGER PRIMARY KEY, data JSON NOT NULL, "
                "version INTEGER NOT NULL, created_at INTEGER, updated_at INTEGER)"
            )
            connection.execute(
                "INSERT INTO config (data, version, created_at, updated_at) VALUES (?, 0, 1, 1)",
                (
                    json.dumps(
                        {
                            "version": 0,
                            "ui": {
                                "banners": [
                                    {
                                        "id": "existing-banner",
                                        "type": "warning",
                                        "content": "用户自己的提醒",
                                    }
                                ],
                                "prompt_suggestions": [
                                    {"title": ["已有建议"], "content": "保留这条建议"}
                                ],
                            },
                            "evaluation": {"arena": {"enable": True}},
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
            connection.commit()
            connection.close()

            result = apply_evidence_ui_config(db_path, timestamp=456)
            self.assertTrue(result["updated"])
            self.assertEqual(result.get("schema"), "blob")

            connection = sqlite3.connect(db_path)
            row = connection.execute("SELECT data FROM config ORDER BY id DESC LIMIT 1").fetchone()
            connection.close()
            payload = json.loads(row[0])
            banners = payload["ui"]["banners"]
            suggestions = payload["ui"]["prompt_suggestions"]
            self.assertEqual(banners[0]["id"], EVIDENCE_BANNER_ID)
            self.assertIn("existing-banner", {item["id"] for item in banners[1:]})
            self.assertEqual(len(suggestions), len(EVIDENCE_SUGGESTIONS) + 1)
            self.assertFalse(payload["evaluation"]["arena"]["enable"])


if __name__ == "__main__":
    unittest.main()
