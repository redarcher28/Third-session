from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kb.store import EvidenceStore
from src.tracks.pipeline import ask

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    store = EvidenceStore()
    print(f"store_count={store.count()}")
    if store.count() == 0:
        print("Knowledge base empty. Run: python scripts/build_kb.py --skip-live")
        sys.exit(1)

    samples = [
        ("clinical", "高血压患者为什么有时要长期吃药？有哪些指南或研究依据？"),
        ("nutrition", "地中海饮食对心血管风险有什么证据？"),
    ]
    for track, q in samples:
        resp = ask(q, track=track)
        print("=" * 60)
        print(track, q)
        print("rewritten:", resp.rewritten_query)
        print("refused:", resp.refused, "contexts:", len(resp.contexts))
        print("cite_ok:", resp.citation_check.get("ok"))
        print(resp.answer[:500])
    print("smoke_demo OK")


if __name__ == "__main__":
    main()
