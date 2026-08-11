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

    # --- 向量服务（国内 OpenAI 兼容 embeddings：硅基流动 / 阿里 DashScope / 智谱）---
    # 留空则退化为 BM25 关键词检索（不产生向量噪声）。
    embedding_api_key: str = ""
    embedding_base_url: str = ""

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

    @property
    def embedding_available(self) -> bool:
        """是否配置了真实向量服务（非占位 Key）。"""
        return bool(self.embedding_api_key) and not self.embedding_api_key.startswith("sk-your-key")


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

    has_key = bool(settings.llm_api_key) and not settings.llm_api_key.startswith(
        "sk-your-key"
    )
    offline = not has_key
    if offline:
        issues.append("未配置有效 LLM_API_KEY，将使用离线占位模式（正式演示需配置）")
    if not settings.llm_base_url:
        issues.append("LLM_BASE_URL 为空")
    if not settings.llm_model:
        issues.append("LLM_MODEL 为空")
    if not settings.embedding_model:
        issues.append("EMBEDDING_MODEL 为空")
    if not settings.embedding_available:
        issues.append("未配置 EMBEDDING_API_KEY，向量检索降级为 BM25 关键词模式")

    # 检查数据目录可写
    try:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        probe = settings.chroma_path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        issues.append(f"向量库目录不可写: {settings.chroma_path} ({e})")

    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        issues.append(".env 不存在（当前使用默认配置）")

    return {
        "ok": not issues or (offline and len(issues) <= 2),
        "offline_mode": offline,
        "issues": issues,
    }
