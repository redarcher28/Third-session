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
from src.runtime_config import RuntimeConfigError, load_runtime_config


class Settings(BaseSettings):
    """应用配置项（可用环境变量覆盖，变量名不区分大小写风格由 pydantic 映射）。"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 大模型 ---
    # ``anthropic`` 对应 Anthropic Messages（Claude）；``openai`` 对应
    # OpenAI Chat Completions；``responses`` 对应 OpenAI Responses API。
    llm_api_format: Literal["openai", "anthropic", "responses"] = "openai"
    llm_api_key: str = "sk-your-key-here"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_reasoning_effort: str = ""

    # 主前端由独立的 Open WebUI 进程提供；FastAPI 只保留 /fallback 备用页。
    openwebui_url: str = "http://127.0.0.1:8080/"

    # Responses/Anthropic 模式不依赖 OpenAI-compatible Embeddings。auto 会在
    # 这两种模式下自动使用本地哈希向量，并由 BM25 保证关键词召回；OpenAI
    # Chat Completions 模式仍默认使用远程 Embeddings，保持旧配置兼容。
    embedding_mode: Literal["auto", "local", "openai"] = "auto"
    embedding_model: str = "text-embedding-3-small"
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
    settings = Settings()
    try:
        runtime = load_runtime_config()
    except RuntimeConfigError:
        # 配置页可继续打开，坏文件不会让整个 RAG 服务无法启动；后续可在页面中恢复 .env。
        runtime = None
    if runtime is not None:
        settings = settings.model_copy(update=runtime.as_settings_overrides())
    return settings


def clear_settings_cache() -> None:
    """让前端保存的新配置在当前进程内立即生效。"""

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 【待完善】运行时配置校验（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def validate_runtime_config() -> dict:
    """
    【待完善】检查 .env 关键配置是否可用（API Key、路径可写、模型名等）。

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
    raise NotImplementedError("待队员实现：validate_runtime_config")
