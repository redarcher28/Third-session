# -*- coding: utf-8 -*-
"""
统一 Web 服务：自定义前端 + Open WebUI 适配 + ReAct + RAG API。

启动：
  uvicorn src.app.web_server:app --reload --port 8000
  或 python run_web.py / scripts/start_openwebui.ps1（含 Open WebUI 8080）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.kb_stats import fetch_kb_stats
from src.app.openwebui import OpenAIChatRequest, chat_completions, model_list
from src.app.settings_api import router as settings_router
from src.config import get_settings
from src.kb.health import kb_health_report
from src.models import AskRequest, AskResponse
from src.tracks.eval_bench import run_benchmark
from src.tracks.pipeline import ask
from src.tracks.prompt_profiles import PROMPT_VERSION, public_track_configs

FRONTEND_DIR = ROOT / "frontend"
FALLBACK_STATIC_DIR = Path(__file__).resolve().parent / "static"
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
POSTERS_DIR = FRONTEND_DIR / "posters"

app = FastAPI(
    title="Evidence Assistant Web",
    description="OpenEvidence 风格证据助手 · 自定义前端 + Open WebUI + ReAct + RAG API",
    version="0.3.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _log_kb_health_on_startup() -> None:
    import logging

    report = kb_health_report()
    if report["status"] != "ok":
        logging.getLogger(__name__).warning(
            "Knowledge base degraded at startup: %s (chroma_ok=%s, bm25=%s/%s)",
            report["degraded_reasons"],
            report["chroma"]["chroma_ok"],
            report["retrieval"]["bm25_indexed_count"],
            report["retrieval"]["bm25_cache_total"],
        )

ETHICS = (
    "本系统仅用于学习与演示，不用于真实诊疗，不处理真实患者隐私。"
    "引用请人工复核。"
)


class ChatMessageItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(default="", max_length=12000)


class ChatConsultRequest(BaseModel):
    messages: list[ChatMessageItem] = Field(default_factory=list)
    track: Literal["clinical", "nutrition"] = "clinical"
    use_react: bool = True
    top_k: int = Field(default=5, ge=3, le=8)
    use_live_tools: bool = False

    @field_validator("messages")
    @classmethod
    def _limit_turns(cls, v: list[ChatMessageItem]) -> list[ChatMessageItem]:
        return v[-50:] if len(v) > 50 else v


def _page(name: str) -> FileResponse | JSONResponse:
    path = FRONTEND_DIR / name
    if path.exists():
        return FileResponse(str(path), media_type="text/html; charset=utf-8")
    return JSONResponse({"message": f"{name} not found"}, status_code=404)


@app.get("/posters/{filename}")
def serve_poster(filename: str) -> FileResponse:
    if not filename.endswith(".svg"):
        raise HTTPException(404, "not found")
    path = POSTERS_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"poster {filename} not found")
    return FileResponse(str(path), media_type="image/svg+xml")


@app.get("/brand/logo.svg", include_in_schema=False)
def serve_brand_logo() -> FileResponse:
    """自定义咨询页头像，避免与 Lee 备用页 /assets 静态目录冲突。"""
    path = ASSETS_DIR / "logo.svg"
    if not path.exists():
        raise HTTPException(404, "logo not found")
    return FileResponse(str(path), media_type="image/svg+xml")


@app.get("/")
def home():
    return _page("index.html")


@app.get("/consult")
def consult_page():
    return _page("consult.html")


@app.get("/eval")
def eval_page():
    return _page("eval.html")


@app.get("/desk", include_in_schema=False)
def openwebui_redirect() -> RedirectResponse:
    """快捷跳转到 Open WebUI 主前端（证据台）。"""
    return RedirectResponse(url=get_settings().openwebui_url, status_code=307)


@app.get("/fallback", include_in_schema=False)
def fallback_web_app() -> FileResponse:
    """Lee 分支备用页（无 Open WebUI 时使用）。"""
    return FileResponse(FALLBACK_STATIC_DIR / "index.html")


# --- 健康与配置 ---


@app.get("/health")
def health() -> dict:
    report = kb_health_report()
    return {
        "status": report["status"],
        "degraded_reasons": report["degraded_reasons"],
        "chroma_ok": report["chroma"]["chroma_ok"],
        "store_count": report["chroma"]["store_count"],
        "bm25_indexed_count": report["retrieval"]["bm25_indexed_count"],
        "bm25_cache_total": report["retrieval"]["bm25_cache_total"],
        "bm25_index_complete": report["retrieval"]["bm25_index_complete"],
        "chromadb_version": report["chroma"]["chromadb_version"],
        "ethics": ETHICS,
        "react": True,
        "prompt_version": PROMPT_VERSION,
        "openwebui_url": get_settings().openwebui_url,
    }


@app.get("/config/tracks")
def track_config() -> dict:
    return public_track_configs()


# --- Open WebUI OpenAI-compatible 适配 ---


@app.get("/v1/models")
def openai_models() -> dict:
    return model_list()


@app.post("/v1/chat/completions")
def openai_chat(request: OpenAIChatRequest) -> dict | object:
    return chat_completions(request)


# --- RAG / ReAct / 评测 ---


@app.post("/api/chat")
def api_chat(req: ChatConsultRequest) -> dict:
    raw = [{"role": m.role, "content": (m.content or "").strip()} for m in req.messages if (m.content or "").strip()]
    if not raw or raw[-1]["role"] != "user":
        raise HTTPException(400, "请至少发送一条用户消息，且最后一条须为用户消息。")
    try:
        if req.use_react:
            from src.agent.react_agent import react_chat

            payload = react_chat(
                raw,
                track=req.track,
                top_k=req.top_k,
                use_live_tools=req.use_live_tools,
            )
        else:
            resp = ask(
                raw[-1]["content"],
                track=req.track,
                top_k=req.top_k,
                use_live_tools=req.use_live_tools,
            )
            payload = {
                "reply": resp.answer,
                "steps": [],
                "contexts": [c.model_dump() for c in resp.contexts],
                "citations": [c.model_dump() for c in resp.citations],
                "track": resp.track,
                "citation_check": resp.citation_check,
                "refused": resp.refused,
                "rewritten_query": resp.rewritten_query,
                "retrieval": resp.retrieval,
                "prompt_version": resp.prompt_version,
            }
    except Exception as e:
        raise HTTPException(500, f"chat failed: {e}") from e
    reply = payload.get("reply") or ""
    if not reply:
        raise HTTPException(502, "模型返回为空。")
    return payload


@app.post("/ask", response_model=AskResponse)
def api_ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(400, "question is empty")
    return ask(
        req.question.strip(),
        track=req.track,
        top_k=req.top_k,
        use_live_tools=req.use_live_tools,
    )


@app.post("/ask/batch", response_model=list[AskResponse])
def api_ask_batch_route(questions: list[AskRequest]) -> list[AskResponse]:
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


@app.post("/eval/run")
def api_eval_run() -> dict:
    return run_benchmark()


@app.get("/eval/results")
def api_eval_results() -> dict:
    path = ROOT / "data" / "eval" / "results" / "benchmark_results.json"
    if not path.exists():
        raise HTTPException(404, "尚无评测结果，请先运行评测。")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/kb/stats")
def kb_stats_route() -> dict:
    return fetch_kb_stats()


if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
if FALLBACK_STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FALLBACK_STATIC_DIR)), name="fallback_assets")
elif ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.include_router(settings_router)
