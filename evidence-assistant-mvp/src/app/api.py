# -*- coding: utf-8 -*-
"""
FastAPI 服务入口。

接口:
- GET  /health      健康检查
- POST /ask         赛道一/二问答
- POST /eval/run    触发赛道三评测
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import AskRequest, AskResponse
from src.kb.store import EvidenceStore
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


@app.post("/ask/batch", response_model=list[AskResponse])
def api_ask_batch(questions: list[AskRequest]) -> list[AskResponse]:
    """
    批量问答接口（评测预跑/压测演示用）。

    请求体:
        list[AskRequest]: 与 /ask 相同的请求结构数组。

    返回:
        list[AskResponse]: 与输入顺序一致的回答列表（空问题跳过）。
    """
    items = [q for q in questions if q.question and q.question.strip()]
    if not items:
        raise HTTPException(400, "all questions are empty")
    return [
        ask(
            q.question.strip(),
            track=q.track,
            top_k=q.top_k,
            use_live_tools=q.use_live_tools,
        )
        for q in items
    ]


@app.post("/eval/run")
def api_eval_run() -> dict:
    """触发完整评测，返回 summary + results。"""
    return run_benchmark()


@app.get("/kb/stats")
def api_kb_stats() -> dict:
    """
    返回当前知识库规模与构成统计。

    参数:
        无

    返回:
        dict: 建议对接 export_store_stats 的结果。

    作用:
        给前端/报告提供「库里有什么」的只读接口。
    """
    store = EvidenceStore()
    chunks = store.all_chunks_for_bm25(limit=5000)
    count = len(chunks)
    by_source: dict[str, int] = collections.Counter()
    by_level: dict[str, int] = collections.Counter()
    tag_counter: collections.Counter = collections.Counter()
    years: list[int] = []
    for c in chunks:
        src = str(c.get("source") or "unknown")
        lvl = str(c.get("evidence_level") or "other")
        by_source[src] += 1
        by_level[lvl] += 1
        raw_tags = c.get("tags")
        if isinstance(raw_tags, str):
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        else:
            tags = [str(t) for t in (raw_tags or []) if t]
        tag_counter.update(tags)
        try:
            year = int(c.get("year"))
            if year > 0:
                years.append(year)
        except (TypeError, ValueError):
            pass
    return {
        "count": count,
        "by_source": dict(by_source),
        "by_level": dict(by_level),
        "top_tags": tag_counter.most_common(10),
        "year_range": [min(years), max(years)] if years else None,
        "collection": "evidence_chunks",
    }
