# -*- coding: utf-8 -*-
"""Open WebUI 使用的 OpenAI-compatible 适配层。

Open WebUI 作为现成前端时，只需要一个能返回模型列表和 Chat Completions
的后端。本模块把 OpenAI 协议请求转换为现有的赛道一/二 ``ask`` 流程，
不复制或改动 A 组的采集、知识库代码。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.models import AskResponse, Citation
from src.tracks.pipeline import ask


logger = logging.getLogger(__name__)


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


def _single_line(value: Any, fallback: str = "") -> str:
    """把证据元数据压成一行，避免文献字段破坏 Open WebUI Markdown。"""

    text = str(value or fallback).replace("\r", " ").replace("\n", " ").strip()
    return " ".join(text.split())


def _markdown_label(value: Any, fallback: str = "") -> str:
    """转义证据标题/摘要中的 Markdown 控制字符。"""

    text = _single_line(value, fallback)
    return (
        text.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("*", "\\*")
        .replace("_", "\\_")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _source_footer(
    citations: list[Citation],
    *,
    retrieval: dict[str, Any] | None = None,
    citation_check: dict[str, Any] | None = None,
) -> str:
    """把旧证据台的证据面板信息转成 Open WebUI 可稳定渲染的 Markdown。"""

    retrieval = retrieval or {}
    citation_check = citation_check or {}
    if not citations and not retrieval:
        return ""

    lines = [
        "\n\n---\n### 证据面板",
        "> 下面的来源由本项目 RAG 后端生成；回答中的 `[n]` 与来源卡片编号对应。",
    ]
    retrieved_count = retrieval.get("retrieved_count")
    if retrieved_count is not None:
        lines.append(f"- 本次检索：{retrieved_count} 条证据")
    rewritten_query = _single_line(retrieval.get("rewritten_query"))
    if rewritten_query:
        lines.append(f"- 改写查询：`{rewritten_query}`")
    rewrite_mode = _single_line(retrieval.get("query_reformulation_mode"))
    if rewrite_mode:
        mode_label = {"lexical": "本地词法扩展", "llm": "远程 LLM 改写", "guarded": "安全边界"}.get(
            rewrite_mode, rewrite_mode
        )
        lines.append(f"- 查询改写方式：{mode_label}")
    sources = retrieval.get("sources") or {}
    if isinstance(sources, dict) and sources:
        source_summary = "，".join(
            f"{_single_line(name, 'unknown')} {count} 条" for name, count in sources.items()
        )
        lines.append(f"- 来源分布：{source_summary}")
    levels = retrieval.get("evidence_levels") or {}
    if isinstance(levels, dict) and levels:
        level_summary = "，".join(
            f"{_single_line(name, 'other')} {count} 条" for name, count in levels.items()
        )
        lines.append(f"- 证据等级：{level_summary}")
    timings = retrieval.get("timings_ms") or {}
    if isinstance(timings, dict) and timings:
        timing_labels = {
            "query_reformulation_ms": "查询改写",
            "retrieval_ms": "检索",
            "relevance_check_ms": "相关性检查",
            "generation_ms": "回答生成",
            "citation_validation_ms": "引用校验",
            "safety_guard_ms": "安全边界",
            "total_ms": "总计",
        }
        timing_parts = [
            f"{timing_labels.get(key, key)} {value:.0f} ms"
            for key, value in timings.items()
            if key in timing_labels
        ]
        if timing_parts:
            lines.append(f"- 分阶段耗时：{' · '.join(timing_parts)}")

    if citations:
        lines.extend(
            [
                "",
                "### 证据来源",
                "回答中的 `[n]` 与本次后端检索采用的证据一一对应；摘要用于快速判断，原文仍需人工核对：",
            ]
        )
        for citation in citations:
            title = _markdown_label(citation.title, citation.doc_id or "未命名来源")
            metadata = [
                _markdown_label(citation.evidence_level, "other"),
                _markdown_label(citation.source, "unknown"),
            ]
            if citation.year:
                metadata.append(str(citation.year))
            lines.append(f"#### [{citation.index}] {title}")
            lines.append(f"- 证据等级 / 来源：{' / '.join(metadata)}")
            snippet = _markdown_label(citation.snippet, "暂无摘要片段")
            lines.append(f"- 摘要：{snippet}")
            if citation.url.strip():
                lines.append(f"  原文：[打开原始来源](<{citation.url.strip()}>)")

    if citation_check:
        checked = "已通过" if citation_check.get("ok", True) else "需复核"
        used = citation_check.get("used_brackets") or []
        used_text = "、".join(f"[{number}]" for number in used)
        suffix = f"；本次使用 {used_text}" if used_text else ""
        lines.extend(["", f"### 引用校验", f"- 状态：{checked}{suffix}"])
    return "\n".join(lines)


def _source_event(citations: list[Citation]) -> str:
    """构造 Open WebUI 原生 Sources 事件。

    Open WebUI 会把 ``event.type=source`` 从上游 SSE 转成当前消息的
    ``sources``，其中 ``document`` 与 ``metadata`` 按位置一一对应。这样
    来源可以进入前端的 Sources 面板，回答正文只保留 ``[n]`` 对照编号，
    不必重复打印完整证据卡片。
    """

    if not citations:
        return ""

    documents: list[str] = []
    metadata: list[dict[str, Any]] = []
    for citation in citations:
        title = _single_line(citation.title, citation.doc_id or "未命名来源")
        url = citation.url.strip()
        locator = url or citation.doc_id or f"evidence-{citation.index}"
        documents.append(citation.snippet.strip() or title)
        metadata.append(
            {
                "source": locator,
                "name": f"[{citation.index}] {title}",
                "url": url,
                "doc_id": citation.doc_id,
                "evidence_level": citation.evidence_level,
                "source_type": citation.source,
                "year": citation.year,
            }
        )

    payload = {
        "event": {
            "type": "source",
            "data": {
                "source": {
                    "id": "evidence-assistant-rag",
                    "name": "证据助手 · 本次 RAG 来源",
                    "type": "evidence",
                },
                "document": documents,
                "metadata": metadata,
            },
        }
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _render_rag_answer(result: AskResponse, *, include_sources: bool = True) -> str:
    """渲染回答；流式 Open WebUI 可把来源改走原生 Sources 面板。"""

    answer = result.answer.strip() or "当前没有可展示的回答。"
    contexts = result.contexts or result.citations
    if include_sources and (contexts or result.retrieval):
        answer += _source_footer(
            contexts,
            retrieval=result.retrieval,
            citation_check=result.citation_check,
        )
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
    include_role: bool = True,
) -> Iterator[str]:
    """把完整回答拆成 OpenAI SSE chunk，让 Open WebUI 增量渲染。"""

    # 一次切一小段即可让界面显示流式状态，同时避免逐字请求造成额外开销。
    chunks = [answer[index : index + 96] for index in range(0, len(answer), 96)] or [""]
    for index, chunk in enumerate(chunks):
        delta: dict[str, Any] = {"content": chunk}
        if include_role and index == 0:
            delta["role"] = "assistant"
        yield _stream_chunk(
            response_id=response_id,
            created=created,
            model=model,
            delta=delta,
        )


def _stream_chunk(
    *,
    response_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> str:
    """构造一个符合 Chat Completions SSE 的增量帧。"""

    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_finished(*, response_id: str, created: int, model: str) -> Iterator[str]:
    """发送结束 chunk 和 OpenAI SSE 的终止标记。"""

    yield _stream_chunk(
        response_id=response_id,
        created=created,
        model=model,
        delta={},
        finish_reason="stop",
    )
    yield "data: [DONE]\n\n"


def _stream_rag_answer(
    *,
    response_id: str,
    created: int,
    model: str,
    question: str,
    track: str,
) -> Iterator[str]:
    """在 RAG worker 中执行问答，并透传真实模型文本增量。"""

    # 先发合法 data 帧；只发 SSE 注释会被部分 Open WebUI/代理忽略，表现为
    # 页面一直等待。后续心跳仍使用注释，避免把状态文案混进回答正文。
    yield _stream_chunk(
        response_id=response_id,
        created=created,
        model=model,
        delta={"role": "assistant"},
    )
    yield ": evidence-assistant retrieval-started\n\n"

    events: Queue[tuple[str, Any]] = Queue()
    stopped = Event()

    def on_text(text: str) -> None:
        if text and not stopped.is_set():
            events.put(("delta", text))

    def run_rag() -> None:
        try:
            events.put(
                (
                    "result",
                    ask(
                        question,
                        track=track,
                        top_k=5,
                        use_live_tools=False,
                        stream_callback=on_text,
                    ),
                )
            )
        except Exception as exc:  # pragma: no cover - 仅保护断开的服务线程
            events.put(("error", exc))
        finally:
            events.put(("done", None))

    Thread(target=run_rag, name="evidence-rag-stream", daemon=True).start()
    result: AskResponse | None = None
    streamed_text = ""
    worker_error: Exception | None = None

    try:
        while True:
            try:
                kind, value = events.get(timeout=15)
            except Empty:
                yield ": evidence-assistant heartbeat\n\n"
                continue
            if kind == "delta":
                text = str(value)
                streamed_text += text
                yield _stream_chunk(
                    response_id=response_id,
                    created=created,
                    model=model,
                    delta={"content": text},
                )
            elif kind == "result":
                result = value
            elif kind == "error":
                worker_error = value
            elif kind == "done":
                break

        if result is None:
            message = "当前问答流被后端中断，请重试。"
            if worker_error:
                logger.warning("Open WebUI streaming worker failed: %s", type(worker_error).__name__)
            yield _stream_chunk(
                response_id=response_id,
                created=created,
                model=model,
                delta={"content": message if not streamed_text else f"\n\n> {message}"},
            )
            yield from _stream_finished(response_id=response_id, created=created, model=model)
            return

        contexts = result.contexts or result.citations
        source_event = _source_event(contexts)
        if source_event:
            yield source_event

        final_answer = _render_rag_answer(result, include_sources=False)
        if not streamed_text:
            yield from _stream_payloads(
                response_id=response_id,
                created=created,
                model=model,
                answer=final_answer,
                include_role=False,
            )
        elif final_answer.startswith(streamed_text):
            remainder = final_answer[len(streamed_text) :]
            if remainder:
                yield _stream_chunk(
                    response_id=response_id,
                    created=created,
                    model=model,
                    delta={"content": remainder},
                )
        elif final_answer.strip() != streamed_text.strip():
            # 流式输出已经不可回退；把差异明确告知用户，Sources 仍以最终
            # 后端校验结果为准，避免悄悄丢失安全提示。
            yield _stream_chunk(
                response_id=response_id,
                created=created,
                model=model,
                delta={"content": "\n\n> 后端已完成引用校验，详细来源见 Sources。"},
            )
        yield from _stream_finished(response_id=response_id, created=created, model=model)
    finally:
        stopped.set()


def chat_completions(request: OpenAIChatRequest) -> dict[str, Any] | StreamingResponse:
    """将 Open WebUI 请求桥接到统一赛道问答链路。"""

    track = _resolve_track(request.model)
    question = _latest_user_question(request.messages)
    if not question:
        raise HTTPException(status_code=400, detail="messages must include a user question")

    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            _stream_rag_answer(
                response_id=response_id,
                created=created,
                model=request.model,
                question=question,
                track=track,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 非流式调用仍需等待完整 RAG 结果；Open WebUI 默认使用上面的流式分支。
    result = ask(question, track=track, top_k=5, use_live_tools=False)
    return _completion_payload(
        response_id=response_id,
        created=created,
        model=request.model,
        answer=_render_rag_answer(result),
    )
