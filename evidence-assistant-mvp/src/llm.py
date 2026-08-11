# -*- coding: utf-8 -*-
"""
大模型客户端封装（OpenAI 兼容协议）。

统一提供：
- chat：文本生成（改写 / 重排 / 回答 / Wiki）
- embed：向量化（检索入库与查询）

未配置有效 API Key 时进入离线占位模式，保证演示链路可跑通。
"""

from __future__ import annotations

import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """轻量聊天 + Embedding 客户端。"""

    def __init__(self) -> None:
        """根据 Settings 初始化远端客户端，并判断是否离线模式。"""
        settings = get_settings()
        self.model = settings.llm_model
        self.embedding_model = settings.embedding_model
        self._client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self._offline = (
            not settings.llm_api_key
            or settings.llm_api_key.startswith("sk-your-key")
        )

    @property
    def is_offline(self) -> bool:
        """是否处于离线占位模式。"""
        return self._offline

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
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

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
        if self._offline:
            return [self._hash_embed(t) for t in texts]
        resp = self._client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        data = sorted(resp.data, key=lambda x: x.index)
        return [list(d.embedding) for d in data]

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
