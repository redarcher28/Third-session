# -*- coding: utf-8 -*-
"""atomic_publish_chunks 校验与 staging 探针失败回滚测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.kb.store import BuildValidationError, atomic_publish_chunks, validate_build_chunks
from src.models import Chunk


def _sample_chunks(n: int = 120) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"chunk-{i}",
            doc_id=f"doc-{i // 10}",
            title=f"Sample title {i}",
            text=f"sample evidence text number {i} hypertension guideline",
            source="local",
            evidence_level="guideline",
            citation_eligible=True,
        )
        for i in range(n)
    ]


class AtomicPublishTests(unittest.TestCase):
    def test_validate_build_chunks_rejects_too_few(self) -> None:
        with self.assertRaises(BuildValidationError):
            validate_build_chunks(_sample_chunks(5), previous_count=1000)

    def test_staging_probe_failure_raises_without_publishing(self) -> None:
        chunks = _sample_chunks(120)

        def fake_probe(store):
            name = getattr(store, "_collection_name", "")
            if "staging" in name:
                return {"chroma_ok": False, "store_count": 0}
            return {"chroma_ok": True, "store_count": len(chunks)}

        with patch("src.kb.health.probe_chroma", side_effect=fake_probe):
            with self.assertRaises(BuildValidationError):
                atomic_publish_chunks(chunks, previous_count=len(chunks))


if __name__ == "__main__":
    unittest.main()
