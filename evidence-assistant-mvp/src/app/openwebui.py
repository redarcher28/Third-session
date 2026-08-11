# -*- coding: utf-8 -*-
"""Open WebUI 使用的 OpenAI-compatible 适配层。

Open WebUI 作为现成前端时，只需要一个能返回模型列表和 Chat Completions
的后端。本模块把 OpenAI 协议请求转换为现有的赛道一/二 ``ask`` 流程，
不复制或改动 A 组的采集、知识库代码。
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.models import AskResponse, Citation
from src.tracks.pipeline import ask


MODEL_PROFILES: dict[str, dict[str, str]] = {
    "evidence-clinical": {
        "track": "clinical",
        "name": "赛道一 · 临床证据",
        "description": "指南与 RCT 优先的结构化证据回答",
    },
    "evidence-nutrition": {
        "track": "nutrition",
        "name": "赛道二 · 健康营养",
        "description": "生活方式与营养证据的通俗解释",
    },
}


class OpenAIMessage(BaseModel):
    """兼容 OpenAI Chat Completions 的最小消息结构。"""

    role: str
    content: Any = ""


class OpenAIChatRequest(BaseModel):
    """Open WebUI 会发送的请求字段；多余字段由 Pydantic 忽略。"""

    model: str = "evidence-clinical"
    messages: list[OpenAIMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None


def model_list() -> dict[str, Any]:
    """返回 Open WebUI 模型选择器需要的模型清单。"""

    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "evidence-assistant",
                "name": profile["name"],
                "description": profile["description"],
            }
            for model_id, profile in MODEL_PROFILES.items()
        ],
    }


def _content_to_text(content: Any) -> str:
    """提取 OpenAI 文本消息；兼容 Open WebUI 的多模态 content 数组。"""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type", "text") == "text":
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _latest_user_question(messages: list[OpenAIMessage]) -> str:
    """从 Open WebUI 的会话消息中取本轮最后一个用户问题。"""

    for message in reversed(messages):
        if message.role.lower() == "user":
            question = _content_to_text(message.content).strip()
            if question:
                return question
    return ""


def _resolve_track(model: str) -> str:
    """把模型选择映射为现有的两个后端赛道。"""

    if model in MODEL_PROFILES:
        return MODEL_PROFILES[model]["track"]
    # 允许 Open WebUI/代理在模型 ID 前加 provider 前缀。
    normalized = model.lower()
    if "nutrition" in normalized or "营养" in model:
        return "nutrition"
    if "clinical" in normalized or "临床" in model:
        return "clinical"
    raise HTTPException(
        status_code=400,
        detail=(
            f"unsupported model: {model}; choose one of "
            f"{', '.join(MODEL_PROFILES)}"
        ),
    )


def _source_footer(citations: list[Citation]) -> str:
    """把后端 RAG 的证据映射显示在 Open WebUI 的回答末尾。"""

    if not citations:
        return ""
    lines = [
        "\n\n---\n### 证据来源",
        "回答中的 `[n]` 与本次后端检索采用的证据一一对应：",
    ]
    for citation in citations:
        title = citation.title.strip() or citation.doc_id.strip() or "未命名来源"
        metadata = [citation.evidence_level or "other", citation.source or "unknown"]
        if citation.year:
            metadata.append(str(citation.year))
        lines.append(f"- [{citation.index}] {title}（{'，'.join(metadata)}）")
        if citation.url.strip():
            lines.append(f"  原文：{citation.url.strip()}")
    return "\n".join(lines)


def _render_rag_answer(result: AskResponse) -> str:
    """渲染回答，并保留从回答到检索证据的可见追踪链。"""

    answer = result.answer.strip() or "当前没有可展示的回答。"
    if result.citations and "### 证据来源" not in answer:
        answer += _source_footer(result.citations)
    return answer


def _completion_payload(
    *,
    response_id: str,
    created: int,
    model: str,
    answer: str,
) -> dict[str, Any]:
    """构造非流式 Chat Completions 响应。"""

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        # 当前 MVP 不暴露供应商 token 计费；保留协议字段，便于 Open WebUI 正常显示。
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _stream_payloads(
    *,
    response_id: str,
    created: int,
    model: str,
    answer: str,
) -> Iterator[str]:
    """把完整回答拆成 OpenAI SSE chunk，让 Open WebUI 增量渲染。"""

    # 一次切一小段即可让界面显示流式状态，同时避免逐字请求造成额外开销。
    chunks = [answer[index : index + 96] for index in range(0, len(answer), 96)] or [""]
    for index, chunk in enumerate(chunks):
        delta: dict[str, Any] = {"content": chunk}
        if index == 0:
            delta["role"] = "assistant"
        payload = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    finished = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(finished, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def chat_completions(request: OpenAIChatRequest) -> dict[str, Any] | StreamingResponse:
    """将 Open WebUI 请求桥接到统一赛道问答链路。"""

    track = _resolve_track(request.model)
    question = _latest_user_question(request.messages)
    if not question:
        raise HTTPException(status_code=400, detail="messages must include a user question")

    # Open WebUI 的模型选择决定赛道；证据数量和在线补检索仍由 B 组服务端统一控制。
    # Open WebUI 自己的 RAG 在启动参数中关闭，避免同一问题被重复检索；这里的 ask()
    # 仍会执行项目知识库的 query reformulation → HybridRetriever → grounded synthesis。
    result = ask(question, track=track, top_k=5, use_live_tools=False)
    answer = _render_rag_answer(result)
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            _stream_payloads(
                response_id=response_id,
                created=created,
                model=request.model,
                answer=answer,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return _completion_payload(
        response_id=response_id,
        created=created,
        model=request.model,
        answer=answer,
    )
