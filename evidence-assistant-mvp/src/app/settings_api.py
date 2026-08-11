# -*- coding: utf-8 -*-
"""本机模型连接设置页与运行时配置 API。

该路由只面向默认的本机开发/演示服务：令牌永远不会通过状态接口返回，
也不会写入仓库。后端默认绑定 127.0.0.1；若改成对外监听，应自行增加反向代理认证。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import clear_settings_cache, get_settings
from src.llm import reset_llm
from src.runtime_config import (
    RuntimeConfigError,
    RuntimeLLMConfig,
    clear_runtime_config,
    load_runtime_config,
    save_runtime_config,
)


router = APIRouter()
STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_API_FORMATS = {"openai", "anthropic", "responses"}
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


class ModelConnectionUpdate(BaseModel):
    """设置页提交的模型连接配置。api_key 为空时保留当前令牌。"""

    api_format: Literal["openai", "anthropic", "responses"] | None = None
    base_url: str | None = Field(default=None, max_length=500)
    model: str | None = Field(default=None, max_length=200)
    reasoning_effort: str | None = Field(default=None, max_length=40)
    api_key: str | None = Field(default=None, max_length=1000)
    clear_api_key: bool = False


def _assert_local_request(request: Request) -> None:
    """阻止默认本机设置接口被远程主机直接改写。"""

    host = request.client.host if request.client else ""
    if host not in LOCAL_HOSTS:
        raise HTTPException(status_code=403, detail="模型连接设置只允许本机访问")


def _is_placeholder_key(value: str) -> bool:
    normalized = value.strip().lower()
    return (
        not normalized
        or normalized.startswith("sk-your-key")
        or normalized.startswith("fill-your-")
        or normalized.startswith("replace-with-")
        or normalized in {"your-api-key", "your-agentrouter-token", "changeme"}
    )


def _mask_key(value: str) -> str:
    """只展示末四位，避免页面状态中泄露真实令牌。"""

    stripped = value.strip()
    if not stripped or _is_placeholder_key(stripped):
        return "未配置"
    return f"••••••••{stripped[-4:]}"


def _validate_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=422, detail="Base URL 必须是 http(s):// 开头的地址")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=422, detail="Base URL 不要包含账号或密码")
    if parsed.query or parsed.fragment:
        raise HTTPException(status_code=422, detail="Base URL 不要包含 query 或 fragment")
    return base_url


def _models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"


def _auth_headers(api_format: str, api_key: str) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if api_format == "anthropic":
        headers["x-api-key"] = api_key
    return headers


def _settings_status() -> dict[str, Any]:
    settings = get_settings()
    try:
        runtime = load_runtime_config()
    except RuntimeConfigError:
        runtime = None
    key = settings.llm_api_key.strip()
    source = "runtime" if runtime is not None else "env" if not _is_placeholder_key(key) else "none"
    return {
        "api_format": settings.llm_api_format,
        "base_url": settings.llm_base_url,
        "model": settings.llm_model,
        "reasoning_effort": settings.llm_reasoning_effort,
        "api_key_configured": not _is_placeholder_key(key),
        "api_key_hint": _mask_key(key),
        "source": source,
        "runtime_storage": "本机私有配置文件（不写入仓库）",
        "openwebui_url": settings.openwebui_url,
    }


@router.get("/settings", include_in_schema=False)
def settings_page() -> FileResponse:
    """提供独立设置页，避免修改 Open WebUI 上游源码。"""

    return FileResponse(STATIC_DIR / "settings.html")


@router.get("/api/settings/status")
def settings_status(request: Request) -> dict[str, Any]:
    _assert_local_request(request)
    return _settings_status()


@router.post("/api/settings/update")
def update_settings(request: Request, payload: ModelConnectionUpdate) -> dict[str, Any]:
    _assert_local_request(request)
    current = get_settings()
    api_format = payload.api_format or current.llm_api_format
    if api_format not in ALLOWED_API_FORMATS:
        raise HTTPException(status_code=422, detail="不支持的 API 格式")

    base_url = _validate_base_url(payload.base_url or current.llm_base_url)
    model = (payload.model or current.llm_model).strip()
    if not model:
        raise HTTPException(status_code=422, detail="模型名不能为空")
    reasoning_effort = (
        payload.reasoning_effort
        if payload.reasoning_effort is not None
        else current.llm_reasoning_effort
    ).strip()

    if payload.clear_api_key:
        api_key = ""
    elif payload.api_key is not None and payload.api_key.strip():
        api_key = payload.api_key.strip()
    else:
        api_key = current.llm_api_key.strip()
    if _is_placeholder_key(api_key):
        api_key = ""

    save_runtime_config(
        RuntimeLLMConfig(
            api_format=api_format,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
        )
    )
    clear_settings_cache()
    reset_llm()
    return {"ok": True, "status": _settings_status()}


@router.post("/api/settings/reset")
def reset_settings(request: Request) -> dict[str, Any]:
    """删除前端保存的覆盖，恢复项目 ``.env`` / 环境变量配置。"""

    _assert_local_request(request)
    clear_runtime_config()
    clear_settings_cache()
    reset_llm()
    return {"ok": True, "status": _settings_status()}


@router.post("/api/settings/test")
def test_settings(request: Request) -> dict[str, Any]:
    """只请求模型列表验证地址/令牌，不发送真实聊天内容。"""

    _assert_local_request(request)
    settings = get_settings()
    api_key = settings.llm_api_key.strip()
    if _is_placeholder_key(api_key):
        return {"ok": False, "message": "尚未配置有效 API Key"}

    try:
        response = httpx.get(
            _models_url(settings.llm_base_url),
            headers=_auth_headers(settings.llm_api_format, api_key),
            timeout=20.0,
        )
    except httpx.HTTPError:
        return {"ok": False, "message": "无法连接到该 API 地址"}

    if response.status_code < 200 or response.status_code >= 300:
        message = f"供应商返回 HTTP {response.status_code}"
        if response.status_code in {401, 403}:
            message = "API Key 未通过认证"
        elif response.status_code >= 500:
            message = f"供应商暂时不可用（HTTP {response.status_code}）"
        return {"ok": False, "status_code": response.status_code, "message": message}

    try:
        data = response.json()
    except ValueError:
        return {"ok": False, "status_code": response.status_code, "message": "供应商返回不是有效 JSON"}
    model_ids = [
        item.get("id")
        for item in data.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]
    return {
        "ok": True,
        "status_code": response.status_code,
        "model_visible": settings.llm_model in model_ids,
        "available_model_count": len(model_ids),
        "message": "连接成功，模型可用" if settings.llm_model in model_ids else "连接成功，但当前模型不在模型列表中",
    }
