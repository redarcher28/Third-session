from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.install_openwebui_bridge import MARKER, build_loader, install_bridge


class OpenWebUIBridgeTests(unittest.TestCase):
    def test_loader_bundles_bridge_without_exposing_api_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bridge_dir = Path(temp_dir)
            (bridge_dir / "openwebui_bridge.css").write_text(".bridge { color: teal; }", encoding="utf-8")
            (bridge_dir / "openwebui_bridge.js").write_text(
                'fetch("__EVIDENCE_BACKEND_URL_VALUE__/api/settings/status");',
                encoding="utf-8",
            )

            loader = build_loader(bridge_dir, "http://127.0.0.1:8000")

        self.assertIn(MARKER, loader)
        self.assertIn(".bridge { color: teal; }", loader)
        self.assertIn("http://127.0.0.1:8000/api/settings/status", loader)
        self.assertNotIn("sk-", loader)

    def test_install_is_idempotent_and_preserves_existing_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frontend_dir = root / "frontend"
            (frontend_dir / "static").mkdir(parents=True)
            (frontend_dir / "static" / "loader.js").write_text(
                "window.existingLoader = true;", encoding="utf-8"
            )
            (root / "openwebui_bridge.css").write_text(".bridge {}", encoding="utf-8")
            (root / "openwebui_bridge.js").write_text("window.bridge = true;", encoding="utf-8")

            first = install_bridge(
                frontend_dir,
                bridge_dir=root,
                backend_url="http://127.0.0.1:8000",
            )
            installed = (frontend_dir / "static" / "loader.js").read_text(encoding="utf-8")
            second = install_bridge(
                frontend_dir,
                bridge_dir=root,
                backend_url="http://127.0.0.1:8000",
            )

            backup = (frontend_dir / "static" / "loader.js.evidence-original").read_text(encoding="utf-8")

        self.assertTrue(first["updated"])
        self.assertTrue(first["backup_created"])
        self.assertFalse(second["backup_created"])
        self.assertIn(MARKER, installed)
        self.assertIn("window.bridge = true;", installed)
        self.assertEqual(backup, "window.existingLoader = true;")


if __name__ == "__main__":
    unittest.main()
