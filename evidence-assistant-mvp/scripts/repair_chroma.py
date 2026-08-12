#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safely rebuild a Chroma index into an isolated directory.

This tool deliberately never modifies ``data/chroma``.  It is intended for a
broken HNSW index where the processed documents remain usable, but the
persisted Chroma files cannot be read.  Build the replacement, inspect the
report, and promote it manually only after an explicit backup/cutover plan.

Example (from the project root)::

    .venv\\Scripts\\python.exe scripts\\repair_chroma.py \\
      --chroma-dir data\\chroma-repair-20260812 \\
      --processed-dir data\\processed-repair-20260812
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild Chroma into a new directory; never replaces the active index."
    )
    parser.add_argument(
        "--source-documents",
        default="data/processed/documents_with_wiki.json",
        help="Existing processed document snapshot to index.",
    )
    parser.add_argument(
        "--chroma-dir",
        required=True,
        help="New, non-existent Chroma directory.",
    )
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="New, non-existent directory for the rebuilt BM25 cache and manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = _path(args.source_documents)
    chroma_dir = _path(args.chroma_dir)
    processed_dir = _path(args.processed_dir)
    active_chroma = (ROOT / "data" / "chroma").resolve()

    if not source.is_file():
        raise SystemExit(f"source document snapshot does not exist: {source}")
    if chroma_dir.resolve() == active_chroma:
        raise SystemExit("refusing to target the active data/chroma directory")
    for target in (chroma_dir, processed_dir):
        if target.exists():
            raise SystemExit(f"refusing to write into an existing path: {target}")

    # These variables are read by Settings after its cache is cleared below.
    # Keep the sidecar cache/manifest isolated as well, so a trial repair does
    # not overwrite the currently-serving BM25 fallback.
    os.environ["CHROMA_DIR"] = str(chroma_dir)
    os.environ["PROCESSED_DIR"] = str(processed_dir)

    from src.config import clear_settings_cache
    from src.ingest import load_docs
    from src.kb.chunking import docs_to_chunks, merge_tiny_chunks
    from src.kb.health import kb_health_report, probe_chroma
    from src.kb.store import EvidenceStore, atomic_publish_chunks
    from src.llm import reset_llm

    clear_settings_cache()
    reset_llm()
    docs = load_docs(source)
    chunks = merge_tiny_chunks(docs_to_chunks(docs))
    if not chunks:
        raise SystemExit("source snapshot produced zero chunks; nothing was written")

    # All writes now target the newly requested paths.  atomic_publish_chunks
    # also validates traceability before it creates a collection.
    written = atomic_publish_chunks(chunks, previous_count=None)
    store = EvidenceStore()
    probe = probe_chroma(store)
    health = kb_health_report()
    report = {
        "source_documents": str(source),
        "chroma_dir": str(chroma_dir),
        "processed_dir": str(processed_dir),
        "document_count": len(docs),
        "chunk_count": len(chunks),
        "written": written,
        "probe": probe,
        "health": health,
    }
    report_path = processed_dir / "repair_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if written != len(chunks) or not probe["chroma_ok"] or probe["store_count"] != written:
        raise SystemExit("isolated rebuild failed verification; active data/chroma was not touched")
    print(f"Verified replacement index. Review: {report_path}")
    print("No active index was replaced. Back up and promote manually only after approval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
