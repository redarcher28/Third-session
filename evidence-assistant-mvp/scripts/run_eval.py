from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tracks.eval_bench import run_benchmark

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG vs baseline evaluation")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "eval" / "benchmark.jsonl",
    )
    args = parser.parse_args()
    payload = run_benchmark(path=args.benchmark)
    print(payload["summary"])


if __name__ == "__main__":
    main()
