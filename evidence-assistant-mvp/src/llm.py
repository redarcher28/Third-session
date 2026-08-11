# -*- coding: utf-8 -*-
"""
大模型客户端封装（OpenAI Chat Completions + Responses + Anthropic Messages）。

统一提供：
- chat：文本生成（改写 / 重排 / 回答 / Wiki）
- embed：向量化（检索入库与查询）

未配置有效 API Key 时进入离线占位模式，保证演示链路可跑通。

ByeAPI 的 GPT 配置使用 OpenAI Responses：
``LLM_API_FORMAT=responses``、``LLM_BASE_URL=https://api.byeapi.top``。
Responses/Anthropic 模式默认使用本地哈希向量 + BM25，不需要第二个令牌。
"""

from __future__ import annotations

import logging
import json
from collections.abc import Iterator
from typing import Any

import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """轻量聊天 + Embedding 客户端。"""

    def __init__(self) -> None:
        """根据 Settings 初始化远端客户端，并判断是否离线模式。"""
        settings = get_settings()
        self.api_format = settings.llm_api_format.strip().lower()
        if self.api_format not in {"openai", "anthropic", "responses"}:
            raise ValueError(
                "LLM_API_FORMAT 只能是 openai、responses 或 anthropic；"
                f"当前值：{settings.llm_api_format!r}"
            )

        self._api_key = settings.llm_api_key.strip()
        self.base_url = settings.llm_base_url.strip()
        self.model = settings.llm_model
        self.reasoning_effort = settings.llm_reasoning_effort.strip()
        self.embedding_model = settings.embedding_model
        self._offline = _is_placeholder_key(self._api_key)

        # Chat Completions 继续使用 OpenAI SDK；Responses/Anthropic 走下面
        # 的显式 HTTP 请求，保证 root base URL 能正确拼出 /v1/responses。
        self._client: OpenAI | None = None
        if self.api_format == "openai":
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self.base_url,
            )

        self.embedding_mode = settings.embedding_mode.strip().lower()
        if self.embedding_mode == "auto":
            self.embedding_mode = (
                "local"
                if self.api_format in {"anthropic", "responses"}
                or "deepseek" in self.base_url.lower()
                else "openai"
            )
        if self.embedding_mode not in {"local", "openai"}:
            raise ValueError(
                "EMBEDDING_MODE 只能是 auto、local 或 openai；"
                f"当前值：{settings.embedding_mode!r}"
            )

        self._embedding_client: OpenAI | None = None
        if not self._offline and self.embedding_mode == "openai":
            # 只有显式提供独立 Embedding 配置时，非 Chat Completions
            # 模式才会把 Embedding 请求发往另一个 OpenAI-compatible 服务。
            embedding_key = settings.embedding_api_key.strip()
            if not embedding_key and self.api_format == "openai":
                embedding_key = self._api_key
            if _is_placeholder_key(embedding_key):
                logger.warning("Embedding key is not configured; falling back to local vectors")
                self.embedding_mode = "local"
            else:
                embedding_base_url = settings.embedding_base_url.strip()
                if not embedding_base_url:
                    embedding_base_url = (
                        self.base_url
                        if self.api_format == "openai"
                        else "https://api.openai.com/v1"
                    )
                self._embedding_client = OpenAI(
                    api_key=embedding_key,
                    base_url=embedding_base_url,
                )

        self._has_remote_embeddings = self._embedding_client is not None

    @property
    def is_offline(self) -> bool:
        """是否处于离线占位模式。"""
        return self._offline

    @property
    def has_remote_embeddings(self) -> bool:
        """是否可以安全使用 Chroma 的远程向量召回。"""
        return self._has_remote_embeddings

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> str:
        """
        调用聊天模型生成文本。

        参数:
            messages: OpenAI 风格消息列表，元素含 role/content。
            temperature: 采样温度，越低越稳定。
            max_tokens: 最大生成 token 数。

        返回:
            str: 模型回复正文（已 strip）。
        """
        if self._offline:
            return self._offline_chat(messages)
        if self.api_format == "anthropic":
            return self._chat_anthropic(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if self.api_format == "responses":
            return self._chat_responses(messages, max_tokens=max_tokens)
        if self._client is None:  # pragma: no cover - 防御性保护
            raise RuntimeError("OpenAI client is not initialized")
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> Iterator[str]:
        """以文本增量形式调用模型，供 Open WebUI SSE 透传。"""

        if self._offline:
            answer = self._offline_chat(messages)
            if answer:
                yield answer
            return
        if self.api_format == "responses":
            yield from self._stream_responses(messages, max_tokens=max_tokens)
            return
        if self.api_format == "anthropic":
            yield from self._stream_anthropic(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return
        if self._client is None:  # pragma: no cover - 防御性保护
            raise RuntimeError("OpenAI client is not initialized")

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for event in stream:
            choices = event.get("choices") if isinstance(event, dict) else getattr(event, "choices", None)
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") if isinstance(choice, dict) else getattr(choice, "delta", None)
            text = delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None)
            if text:
                yield str(text)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        批量文本向量化。

        参数:
            texts: 待编码文本列表。

        返回:
            list[list[float]]: 与输入顺序一致的向量列表。
        """
        if not texts:
            return []
        if self._offline or self.embedding_mode == "local":
            return [self._hash_embed(t) for t in texts]
        if self._embedding_client is None:  # pragma: no cover - 防御性保护
            raise RuntimeError("Embedding client is not initialized")
        resp = self._embedding_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        data = sorted(resp.data, key=lambda x: x.index)
        return [list(d.embedding) for d in data]

    def _chat_responses(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
    ) -> str:
        """调用 OpenAI Responses-compatible 接口，并提取 output_text。"""
        payload: dict[str, object] = {
            "model": self.model,
            "input": messages,
            "max_output_tokens": max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        response = httpx.post(
            self._responses_url(),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "content-type": "application/json",
            },
            json=payload,
            timeout=120.0,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text[:500].replace("\n", " ")
            raise RuntimeError(
                f"Responses API 请求失败（HTTP {response.status_code}）：{detail}"
            ) from exc

        data = response.json()
        answer = str(data.get("output_text") or "").strip()
        if not answer:
            blocks: list[str] = []
            for item in data.get("output", []):
                if not isinstance(item, dict):
                    continue
                for block in item.get("content", []):
                    if isinstance(block, dict) and block.get("type") in {
                        "output_text",
                        "text",
                    }:
                        blocks.append(str(block.get("text") or ""))
            answer = "".join(blocks).strip()
        if not answer:
            raise RuntimeError("Responses API 返回中没有可读文本")
        return answer

    @staticmethod
    def _iter_sse_data(lines: Iterator[str]) -> Iterator[str]:
        """把上游 SSE 行合并为 data payload，兼容多行 data 字段。"""

        data_lines: list[str] = []
        for line in lines:
            line = line.rstrip("\r")
            if not line:
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            yield "\n".join(data_lines)

    @staticmethod
    def _json_stream_text(payload: dict[str, Any]) -> str:
        """从 Responses/兼容网关事件中提取一段可展示文本。"""

        event_type = str(payload.get("type") or "")
        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            return str(payload.get("delta") or "")
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                delta = choice.get("delta") or {}
                if isinstance(delta, dict):
                    return str(delta.get("content") or "")
        # 少数兼容网关会把增量放在 output_text 字段中。
        return str(payload.get("output_text") or "") if payload.get("output_text") else ""

    def _stream_responses(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
    ) -> Iterator[str]:
        """消费 Responses API 的 response.output_text.delta 事件。"""

        payload: dict[str, object] = {
            "model": self.model,
            "input": messages,
            "max_output_tokens": max_tokens,
            "stream": True,
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        with httpx.stream(
            "POST",
            self._responses_url(),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "content-type": "application/json",
                "accept": "text/event-stream",
            },
            json=payload,
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            for raw in self._iter_sse_data(response.iter_lines()):
                if raw.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                text = self._json_stream_text(event)
                if text:
                    yield text

    def _stream_anthropic(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        """消费 Anthropic Messages 的 content_block_delta 事件。"""

        system_parts = [
            message["content"]
            for message in messages
            if message.get("role") == "system" and message.get("content")
        ]
        conversation = [
            {
                "role": message.get("role", "user"),
                "content": message.get("content", ""),
            }
            for message in messages
            if message.get("role") in {"user", "assistant"}
        ]
        if not conversation:
            conversation = [{"role": "user", "content": ""}]

        payload: dict[str, object] = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        with httpx.stream(
            "POST",
            self._anthropic_messages_url(),
            headers={
                "x-api-key": self._api_key,
                "Authorization": f"Bearer {self._api_key}",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "accept": "text/event-stream",
            },
            json=payload,
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            for raw in self._iter_sse_data(response.iter_lines()):
                if raw.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                text = ""
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta") or {}
                    if isinstance(delta, dict) and delta.get("type") == "text_delta":
                        text = str(delta.get("text") or "")
                elif event.get("type") == "completion":
                    text = str(event.get("completion") or "")
                if text:
                    yield text

    def _responses_url(self) -> str:
        """根据 provider root/base URL 生成 Responses endpoint。"""
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/responses"
        return f"{base}/v1/responses"

    def _chat_anthropic(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """调用 Anthropic Messages 接口，并把响应转换成纯文本。"""
        system_parts = [
            message["content"]
            for message in messages
            if message.get("role") == "system" and message.get("content")
        ]
        conversation = [
            {
                "role": message.get("role", "user"),
                "content": message.get("content", ""),
            }
            for message in messages
            if message.get("role") in {"user", "assistant"}
        ]
        if not conversation:
            conversation = [{"role": "user", "content": ""}]

        payload: dict[str, object] = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        response = httpx.post(
            self._anthropic_messages_url(),
            headers={
                # x-api-key 是 Anthropic Messages 的标准认证头；额外提供
                # Bearer 兼容头，覆盖 AgentRouter/网关常见的 AUTH_TOKEN 形式。
                "x-api-key": self._api_key,
                "Authorization": f"Bearer {self._api_key}",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=120.0,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text[:500].replace("\n", " ")
            raise RuntimeError(
                f"Anthropic API 请求失败（HTTP {response.status_code}）：{detail}"
            ) from exc

        data = response.json()
        text_blocks = [
            block.get("text", "")
            for block in data.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        answer = "".join(str(block) for block in text_blocks).strip()
        if not answer:
            raise RuntimeError("Anthropic API 返回中没有可读文本")
        return answer

    def _anthropic_messages_url(self) -> str:
        """根据可编辑的 base URL 生成 Messages endpoint。"""
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/messages"
        return f"{base}/v1/messages"

    def _offline_chat(self, messages: list[dict[str, str]]) -> str:
        """
        离线占位回答（无 API Key 时使用）。

        参数:
            messages: 同 chat。

        返回:
            str: 确定性模板文本，保证链路不中断。
        """
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        if "rewrite" in system.lower() or "改写" in system:
            return user.strip().split("\n")[0][:200]
        # 不要用“相关性”作重排识别词：赛道二的正常科普 Prompt 也会讨论
        # 相关性，离线模式应继续返回证据回答，而不是把候选编号泄露到界面。
        if "rerank" in system.lower() or "只输出逗号分隔的候选编号" in system:
            return "1,2,3,4,5"
        if "wiki" in system.lower() or "主题知识页" in system:
            return (
                "# 主题摘要（离线占位）\n\n"
                "本页由离线模式生成，仅作演示结构占位。请配置 LLM_API_KEY 后重建 Wiki。\n"
            )
        if "证据" in system or "citation" in system.lower() or "[" in user:
            return (
                "根据当前检索到的证据，相关研究发现生活方式干预与规范药物治疗在"
                "血脂/血压管理中均有支持依据[1]。证据不足处请结合临床判断，"
                "本系统不提供个体化诊疗建议。\n\n"
                "参考文献见文末列表。"
            )
        return (
            "（离线模式）未配置有效 LLM_API_KEY。请在 .env 中配置后重试。"
            f" 收到问题：{user[:120]}"
        )

    @staticmethod
    def _hash_embed(text: str, dim: int = 384) -> list[float]:
        """
        离线哈希向量（仅用于本地演示，不可替代真实 Embedding）。

        参数:
            text: 输入文本。
            dim: 向量维度。

        返回:
            list[float]: 归一化后的伪向量。
        """
        import hashlib
        import math
        import re

        vec = [0.0] * dim
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
        if not tokens:
            tokens = ["empty"]
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
            vec[(h // dim) % dim] += 0.5
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _is_placeholder_key(value: str) -> bool:
    """识别示例令牌，避免把占位字符串当成真实密钥发到远端。"""
    normalized = value.strip().lower()
    return (
        not normalized
        or normalized.startswith("sk-your-key")
        or normalized.startswith("fill-your-")
        or normalized.startswith("replace-with-")
        or normalized in {"your-api-key", "your-agentrouter-token", "changeme"}
    )


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    """
    获取全局 LLMClient 单例。

    参数:
        无

    返回:
        LLMClient: 可复用的客户端实例。
    """
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def reset_llm() -> None:
    """丢弃旧客户端，使运行时配置保存后下一次请求重新建连。"""

    global _client
    _client = None


# ---------------------------------------------------------------------------
# 【待完善】模型调用增强（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def with_json_mode_chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
) -> dict:
    """
    【待完善】要求模型输出 JSON，并解析为 dict。

    参数:
        messages: 对话消息。
        temperature: 采样温度。

    返回:
        dict: 解析后的 JSON 对象。

    作用:
        供重排、大纲、评分解析等结构化任务复用，减少正则抠文本。
    """
    raise NotImplementedError("待队员实现：with_json_mode_chat")


def embed_with_cache(texts: list[str], cache_dir: str | None = None) -> list[list[float]]:
    """
    【待完善】带本地缓存的 embedding，避免重复计费。

    参数:
        texts: 文本列表。
        cache_dir: 缓存目录；None 时使用默认路径。

    返回:
        list[list[float]]: 与输入对齐的向量。

    作用:
        加快反复建库/调试时的向量化速度。
    """
    raise NotImplementedError("待队员实现：embed_with_cache")
