#!/usr/bin/env python3
"""把证据模型连接卡片注入 Open WebUI 的现有设置入口。

Open WebUI 通过 ``frontend/static/loader.js`` 提供一个稳定的自定义加载点。
启动前把本项目的桥接脚本和样式合并到这个 loader，再由 Open WebUI 自己复制
到运行时静态目录。这样不需要复制或改写 Open WebUI 的 Svelte 前端源码，且
每次启动都可以幂等重放；原有非本项目 loader 会先保留一份旁路备份。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


MARKER = "evidence-assistant-openwebui-bridge-v1"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRIDGE_DIR = PROJECT_ROOT / "src" / "app" / "static"


def _validate_backend_url(value: str) -> str:
    url = value.strip().rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("backend URL 必须是 http(s):// 开头的地址")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("backend URL 不要包含账号、密码、query 或 fragment")
    return url


def package_frontend_dir() -> Path:
    """定位当前 Conda 环境中的 Open WebUI 前端构建目录。"""

    try:
        import open_webui
    except ImportError as exc:  # pragma: no cover - 启动脚本会先做环境检查
        raise RuntimeError("当前环境没有安装 open-webui") from exc
    frontend_dir = Path(open_webui.__file__).resolve().parent / "frontend"
    if not (frontend_dir / "static" / "loader.js").exists():
        raise RuntimeError(f"找不到 Open WebUI loader：{frontend_dir / 'static' / 'loader.js'}")
    return frontend_dir


def build_loader(bridge_dir: Path, backend_url: str) -> str:
    """把项目桥接脚本、样式和本机后端地址组装成 loader.js。"""

    css_path = bridge_dir / "openwebui_bridge.css"
    js_path = bridge_dir / "openwebui_bridge.js"
    css = css_path.read_text(encoding="utf-8")
    bridge_js = js_path.read_text(encoding="utf-8")
    bridge_js = bridge_js.replace("__EVIDENCE_BACKEND_URL_VALUE__", backend_url)
    css_literal = json.dumps(css, ensure_ascii=False)
    backend_literal = json.dumps(backend_url, ensure_ascii=False)
    style_loader = f"""(() => {{
  const installEvidenceBridgeStyle = () => {{
    if (!document.head || document.getElementById("evidence-bridge-style")) return;
    const style = document.createElement("style");
    style.id = "evidence-bridge-style";
    style.textContent = {css_literal};
    document.head.appendChild(style);
  }};
  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", installEvidenceBridgeStyle, {{ once: true }});
  }} else {{
    installEvidenceBridgeStyle();
  }}
}})();
"""
    return (
        f"/* {MARKER} */\n"
        f"window.__EVIDENCE_BACKEND_URL__ = {backend_literal};\n"
        f"{style_loader}\n"
        f"{bridge_js}\n"
    )


def install_bridge(
    frontend_dir: Path,
    *,
    bridge_dir: Path = DEFAULT_BRIDGE_DIR,
    backend_url: str = DEFAULT_BACKEND_URL,
) -> dict[str, object]:
    """幂等安装桥接 loader，返回不包含敏感信息的审计摘要。"""

    backend_url = _validate_backend_url(backend_url)
    loader_path = frontend_dir / "static" / "loader.js"
    loader_path.parent.mkdir(parents=True, exist_ok=True)
    existing = loader_path.read_text(encoding="utf-8") if loader_path.exists() else ""
    backup_path = loader_path.with_name("loader.js.evidence-original")
    backup_created = False
    if existing.strip() and MARKER not in existing and not backup_path.exists():
        backup_path.write_text(existing, encoding="utf-8")
        backup_created = True

    loader = build_loader(bridge_dir, backend_url)
    loader_path.write_text(loader, encoding="utf-8")
    return {
        "updated": True,
        "marker": MARKER,
        "loader": str(loader_path),
        "backup_created": backup_created,
        "backend_host": urlsplit(backend_url).netloc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the Evidence Desk bridge into Open WebUI")
    parser.add_argument("--frontend-dir", type=Path, default=None)
    parser.add_argument("--bridge-dir", type=Path, default=DEFAULT_BRIDGE_DIR)
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("EVIDENCE_BACKEND_URL", DEFAULT_BACKEND_URL),
    )
    args = parser.parse_args()
    frontend_dir = args.frontend_dir or package_frontend_dir()
    result = install_bridge(
        frontend_dir,
        bridge_dir=args.bridge_dir,
        backend_url=args.backend_url,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
