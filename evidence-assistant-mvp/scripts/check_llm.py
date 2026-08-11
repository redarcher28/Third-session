"""检查项目 .env 中配置的模型，且不会打印真实 API Key。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.llm import get_llm


def _models_url(base_url: str) -> str:
    """把可编辑的 provider base URL 转成 OpenAI-style /v1/models 地址。"""
    base = base_url.rstrip("/")
    return f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"


def _auth_headers(api_format: str, api_key: str) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if api_format == "anthropic":
        headers["x-api-key"] = api_key
    return headers


def main() -> int:
    parser = argparse.ArgumentParser(description="检查证据助手的 LLM 配置")
    parser.add_argument(
        "--models",
        action="store_true",
        help="只查询 provider 的模型列表，不发送聊天请求",
    )
    args = parser.parse_args()

    settings = get_settings()
    print(f"api_format={settings.llm_api_format}")
    print(f"base_url={settings.llm_base_url}")
    print(f"model={settings.llm_model}")
    print(f"api_key={'configured' if settings.llm_api_key.strip() else 'missing'}")

    llm = get_llm()
    if llm.is_offline:
        print("未配置有效令牌：当前仍是离线占位模式。")
        return 2

    if args.models:
        response = httpx.get(
            _models_url(settings.llm_base_url),
            headers=_auth_headers(settings.llm_api_format, settings.llm_api_key),
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        model_ids = [item.get("id") for item in data if isinstance(item, dict) and item.get("id")]
        print("available_models=")
        print("\n".join(str(model_id) for model_id in model_ids) or "<empty>")
        return 0

    reply = llm.chat(
        [{"role": "user", "content": "只回复 OK，不要添加其它内容。"}],
        temperature=0,
        max_tokens=16,
    )
    print(f"reply={reply}")
    print("LLM connection OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
