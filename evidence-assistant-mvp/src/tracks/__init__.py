"""赛道模块包；避免在 import 时加载 pipeline，防止与 generation.answer 循环依赖。"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "ask":
        from src.tracks.pipeline import ask

        return ask
    if name == "run_benchmark":
        from src.tracks.eval_bench import run_benchmark

        return run_benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ask", "run_benchmark"]
