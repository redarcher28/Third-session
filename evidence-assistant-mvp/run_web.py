#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyCharm 一键启动 Web 服务（推荐入口）。

用法：
  在 PyCharm 中右键本文件 → Run「run_web」
  或在终端：python run_web.py

启动后约 1 秒自动打开浏览器至首页。
"""

from __future__ import annotations

import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOST = "127.0.0.1"
PORT = 8000
HOME_URL = f"http://{HOST}:{PORT}/"


def _open_browser() -> None:
    try:
        webbrowser.open(HOME_URL)
    except Exception:
        pass


def main() -> None:
    import uvicorn

    posters = ROOT / "frontend" / "posters"
    if not (posters / "clinical-scene.svg").exists():
        print("⚠ 警告：海报文件缺失，请确认 frontend/posters/ 目录存在。")

    print("=" * 58)
    print("  Evidence Assistant · Web 服务")
    print(f"  自定义首页 http://{HOST}:{PORT}/")
    print(f"  临床咨询   http://{HOST}:{PORT}/consult?track=clinical")
    print(f"  营养咨询   http://{HOST}:{PORT}/consult?track=nutrition")
    print(f"  评测面板   http://{HOST}:{PORT}/eval")
    print(f"  Open WebUI http://127.0.0.1:8080/  (需 scripts/start_openwebui.ps1)")
    print(f"  API 适配   http://{HOST}:{PORT}/v1/chat/completions")
    print("  停止服务：PyCharm 点击红色停止按钮，或 Ctrl+C")
    print("=" * 58)

    # 延迟打开，避免 uvicorn 尚未监听时浏览器访问失败
    threading.Timer(1.2, _open_browser).start()

    uvicorn.run(
        "src.app.web_server:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        access_log=True,
    )


if __name__ == "__main__":
    main()
