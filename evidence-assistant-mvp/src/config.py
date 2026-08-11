# -*- coding: utf-8 -*-
"""
全局配置模块。

从环境变量 / .env 读取大模型、路径、NCBI 等配置，供全项目共用。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from src import PROJECT_ROOT


class Settings(BaseSettings):
    """应用配置项（可用环境变量覆盖，变量名不区分大小写风格由 pydantic 映射）。"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 大模型（OpenAI 兼容接口）---
    llm_api_key: str = "sk-your-key-here"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Embedding 独立端点（可选；不填则复用 LLM 配置）---
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_mode: str = "auto"  # auto / api / offline

    # --- NCBI / PubMed 礼貌访问参数 ---
    ncbi_api_key: str = ""
    ncbi_email: str = "evidence-mvp@example.com"

    # --- 相对项目根目录的数据路径 ---
    chroma_dir: str = "data/chroma"
    processed_dir: str = "data/processed"
    raw_dir: str = "data/raw"

    @property
    def chroma_path(self) -> Path:
        """向量库落盘绝对路径。"""
        return PROJECT_ROOT / self.chroma_dir

    @property
    def processed_path(self) -> Path:
        """处理后文档目录绝对路径。"""
        return PROJECT_ROOT / self.processed_dir

    @property
    def raw_path(self) -> Path:
        """原始采集数据目录绝对路径。"""
        return PROJECT_ROOT / self.raw_dir


# 赛道名称字面量，供类型标注使用
TrackName = Literal["clinical", "nutrition"]


@lru_cache
def get_settings() -> Settings:
    """
    获取全局配置单例。

    参数:
        无

    返回:
        Settings: 已加载的配置对象。
    """
    return Settings()


# ---------------------------------------------------------------------------
# 运行时配置校验
# ---------------------------------------------------------------------------


def validate_runtime_config() -> dict:
    """
    检查 .env 关键配置是否可用（API Key、路径可写、模型名等）。

    参数:
        无

    返回:
        dict: 例如 {
            "ok": bool,
            "offline_mode": bool,
            "issues": list[str],
        }

    作用:
        演示前一键自检，避免上台才发现未配 Key 或目录不可写。
    """
    settings = get_settings()
    issues: list[str] = []
    blocking: list[str] = []

    offline_mode = not settings.llm_api_key or settings.llm_api_key.startswith(
        "sk-your-key"
    )
    if offline_mode:
        issues.append("未配置有效 LLM_API_KEY，将使用离线占位模式（链路可跑，回答为占位）。")
    if not settings.llm_model.strip():
        issues.append("LLM_MODEL 为空。")
        blocking.append("llm_model")
    if not settings.embedding_model.strip():
        issues.append("EMBEDDING_MODEL 为空。")
        blocking.append("embedding_model")

    for name, path in [
        ("chroma", settings.chroma_path),
        ("processed", settings.processed_path),
        ("raw", settings.raw_path),
    ]:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            issues.append(f"{name} 目录不可写：{path}（{e}）")
            blocking.append(name)

    if not (settings.processed_path / "documents_with_wiki.json").exists():
        issues.append(
            "缺少 data/processed/documents_with_wiki.json，请先运行 "
            "python scripts/build_kb.py --skip-live。"
        )
        blocking.append("processed_docs")

    return {
        "ok": not blocking,
        "offline_mode": offline_mode,
        "issues": issues,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
    }
