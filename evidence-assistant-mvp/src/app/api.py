# -*- coding: utf-8 -*-
"""
FastAPI 服务入口。

接口:
- GET  /health      健康检查
- POST /ask         赛道一/二问答
- GET  /v1/models   Open WebUI 模型列表
- POST /v1/chat/completions  Open WebUI 对话适配
- POST /eval/run    触发赛道三评测
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import AskRequest, AskResponse
from src.config import get_settings
from src.app.openwebui import (
    OpenAIChatRequest,
    chat_completions,
    model_list,
)
from src.app.settings_api import router as settings_router
from src.kb.store import export_store_stats
from src.tracks.prompt_profiles import PROMPT_VERSION, public_track_configs
from src.tracks.eval_bench import run_benchmark
from src.tracks.pipeline import ask

app = FastAPI(
    title="Evidence Assistant MVP",
    description="OpenEvidence 风格证据助手：临床 / 营养 / 评测",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/assets", StaticFiles(directory=str(STATIC_DIR)), name="assets")
app.include_router(settings_router)

ETHICS = (
    "本系统仅用于学习与演示，不用于真实诊疗，不处理真实患者隐私信息。"
    "引用请人工复核。"
)


@app.get("/health")
def health() -> dict:
    """健康检查，附带伦理声明。"""
    return {"status": "ok", "ethics": ETHICS, "prompt_version": PROMPT_VERSION}


@app.get("/", include_in_schema=False)
def web_app() -> RedirectResponse:
    """把根入口交给真正的 Open WebUI 主前端。"""
    return RedirectResponse(url=get_settings().openwebui_url, status_code=307)


@app.get("/fallback", include_in_schema=False)
def fallback_web_app() -> FileResponse:
    """提供无额外依赖的旧版备用页，不作为主前端。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/config/tracks")
def track_config() -> dict:
    """返回前端需要的赛道文案、示例问题和安全边界。"""
    return public_track_configs()


@app.get("/v1/models")
def openai_models() -> dict:
    """Open WebUI 连接 OpenAI-compatible provider 时读取的模型列表。"""
    return model_list()


@app.post("/v1/chat/completions")
def openai_chat(request: OpenAIChatRequest) -> dict | object:
    """将 Open WebUI 的对话请求转到赛道一/二统一流水线。"""
    return chat_completions(request)


@app.get("/kb/stats")
def kb_stats() -> dict:
    """返回知识库规模与来源分布，供前端状态栏展示。"""
    return api_kb_stats()


@app.post("/ask", response_model=AskResponse)
def api_ask(req: AskRequest) -> AskResponse:
    """
    问答接口。

    请求体 AskRequest:
        question, track, use_live_tools, top_k
    返回:
        AskResponse
    """
    if not req.question.strip():
        raise HTTPException(400, "question is empty")
    return ask(
        req.question.strip(),
        track=req.track,
        top_k=req.top_k,
        use_live_tools=req.use_live_tools,
    )


@app.post("/eval/run")
def api_eval_run() -> dict:
    """触发完整评测，返回 summary + results。"""
    return run_benchmark()


@app.post("/ask/batch", response_model=list[AskResponse])
def api_ask_batch_route(questions: list[AskRequest]) -> list[AskResponse]:
    """统一批量问答接口，供评测预跑和脚本调用。"""
    return api_ask_batch(questions)


# ---------------------------------------------------------------------------
# 【待完善】API 扩展（只定义签名与备注；若作为路由需自行挂载装饰器）
# ---------------------------------------------------------------------------


def api_ask_batch(questions: list[AskRequest]) -> list[AskResponse]:
    """
    【待完善】批量问答接口逻辑（可用于评测预跑或压力演示）。

    参数:
        questions: AskRequest 列表。

    返回:
        list[AskResponse]: 与输入顺序对应的回答列表。

    作用:
        减少逐条 HTTP 往返，方便脚本化评测。
    """
    responses: list[AskResponse] = []
    for req in questions:
        if not req.question.strip():
            raise HTTPException(400, "question is empty")
        responses.append(
            ask(
                req.question.strip(),
                track=req.track,
                top_k=req.top_k,
                use_live_tools=req.use_live_tools,
            )
        )
    return responses

def api_kb_stats() -> dict:
    """
    【待完善】返回当前知识库规模与构成统计。

    参数:
        无

    返回:
        dict: 建议对接 export_store_stats 的结果。

    作用:
        给前端/报告提供「库里有什么」的只读接口。
    """
    try:
        stats = export_store_stats()
    except Exception as exc:
        # 健康接口和前端状态栏不应因知识库尚未构建而无法打开。
        return {
            "ok": False,
            "count": 0,
            "docs_covered": 0,
            "by_source": {},
            "by_level": {},
            "error": str(exc),
        }
    return {"ok": True, **stats}
