from __future__ import annotations

import unittest
from pathlib import Path

from scripts.repair_chroma import ROOT, _path


class RepairChromaPathTests(unittest.TestCase):
    def test_relative_paths_are_rooted_at_project(self) -> None:
        self.assertEqual(_path("data/repair").resolve(), (ROOT / "data" / "repair").resolve())

    def test_absolute_path_is_preserved(self) -> None:
        absolute = (ROOT / "temporary-repair-target").resolve()
        self.assertEqual(_path(str(absolute)), absolute)


if __name__ == "__main__":
    unittest.main()
