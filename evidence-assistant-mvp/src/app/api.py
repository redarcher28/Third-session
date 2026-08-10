# -*- coding: utf-8 -*-
"""
FastAPI 服务入口。

接口:
- GET  /health      健康检查
- POST /ask         赛道一/二问答
- POST /eval/run    触发赛道三评测
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import AskRequest, AskResponse
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

ETHICS = (
    "本系统仅用于学习与演示，不用于真实诊疗，不处理真实患者隐私信息。"
    "引用请人工复核。"
)


@app.get("/health")
def health() -> dict:
    """健康检查，附带伦理声明。"""
    return {"status": "ok", "ethics": ETHICS}


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
    raise NotImplementedError("待队员实现：api_ask_batch")


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
    raise NotImplementedError("待队员实现：api_kb_stats")
