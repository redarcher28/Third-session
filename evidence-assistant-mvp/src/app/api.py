# -*- coding: utf-8 -*-
"""
FastAPI 服务入口（与 web_server 共用同一 app，供 Open WebUI 启动脚本引用）。

接口:
- GET  /health, /config/tracks, /kb/stats
- POST /ask, /ask/batch, /eval/run
- GET  /v1/models, POST /v1/chat/completions  （Open WebUI）
- POST /api/chat  （自定义前端 ReAct）
"""

from src.app.web_server import app

__all__ = ["app"]
