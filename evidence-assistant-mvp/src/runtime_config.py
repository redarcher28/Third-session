# -*- coding: utf-8 -*-
"""本机运行时模型配置。

前端设置页写入独立于仓库的 JSON 文件，避免用户为了切换供应商反复编辑
项目 ``.env``。文件只保存模型连接配置，并在落盘时限制为当前用户可读写。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError


class RuntimeConfigError(RuntimeError):
    """运行时配置文件不可读或格式不正确。"""


class RuntimeLLMConfig(BaseModel):
    """前端可编辑的 LLM 连接字段。"""

    model_config = ConfigDict(extra="ignore")

    api_format: Literal["openai", "anthropic", "responses"]
    api_key: str = ""
    base_url: str
    model: str
    reasoning_effort: str = ""

    def as_settings_overrides(self) -> dict[str, str]:
        """转换成 ``Settings`` 使用的字段名。"""

        return {
            "llm_api_format": self.api_format,
            "llm_api_key": self.api_key,
            "llm_base_url": self.base_url,
            "llm_model": self.model,
            "llm_reasoning_effort": self.reasoning_effort,
        }


def runtime_config_path() -> Path:
    """返回本机运行时配置路径，默认不落在 Git 工作区。"""

    configured = os.environ.get("EVIDENCE_RUNTIME_CONFIG", "").strip()
    if configured:
        return Path(configured).expanduser()
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        root = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return root / "evidence-assistant-mvp" / "llm_runtime.json"


def load_runtime_config() -> RuntimeLLMConfig | None:
    """读取运行时覆盖；没有文件时返回 ``None``。"""

    path = runtime_config_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("llm"), dict):
            payload = payload["llm"]
        return RuntimeLLMConfig.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError) as exc:
        # 不把可能包含令牌的原始 payload 放进异常文本或日志。
        raise RuntimeConfigError("本机运行时模型配置不可读取或格式错误") from exc


def save_runtime_config(config: RuntimeLLMConfig) -> Path:
    """原子写入配置，并尽力设置 macOS/Linux 的用户私有权限。"""

    path = runtime_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {"version": 1, "llm": config.model_dump(mode="json")}
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def clear_runtime_config() -> bool:
    """删除运行时覆盖，恢复使用 ``.env`` 或环境变量。"""

    path = runtime_config_path()
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True
