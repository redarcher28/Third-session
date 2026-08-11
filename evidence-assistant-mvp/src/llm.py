# -*- coding: utf-8 -*-
"""
大模型客户端封装（OpenAI 兼容协议）。

统一提供：
- chat：文本生成（改写 / 重排 / 回答 / Wiki）
- embed：向量化（检索入库与查询）

未配置有效 API Key 时进入离线占位模式，保证演示链路可跑通。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src import PROJECT_ROOT
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
        # 独立向量客户端：国内服务商（硅基流动 / 阿里 DashScope / 智谱等）的
        # OpenAI 兼容 embeddings 接口；未配置时退化为关键词检索。
        self._embed_client = OpenAI(
            api_key=settings.embedding_api_key or settings.llm_api_key,
            base_url=settings.embedding_base_url or settings.llm_base_url,
        )
        self._embedding_available = settings.embedding_available

    @property
    def is_offline(self) -> bool:
        """是否处于离线占位模式。"""
        return self._offline

    @property
    def embedding_available(self) -> bool:
        """是否可调用真实向量服务（False 时混合检索跳过向量分支）。"""
        return self._embedding_available

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
        if not self._embedding_available:
            # 未配置独立向量服务：直接哈希占位，避免对聊天服务商发无效请求。
            return [self._hash_embed(t) for t in texts]
        try:
            resp = self._embed_client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
            data = sorted(resp.data, key=lambda x: x.index)
            return [list(d.embedding) for d in data]
        except Exception as e:
            # 部分服务商（如 DeepSeek）不提供 embedding 接口：
            # 降级为本地哈希向量，保证检索链路不中断。
            logger.warning("embedding API failed (%s); fallback to hash embedding", e)
            return [self._hash_embed(t) for t in texts]

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
        if "rerank" in system.lower() or "相关性" in system:
            return "1,2,3,4,5"
        if "wiki" in system.lower() or "主题知识页" in system:
            return (
                "# 主题摘要（离线占位）\n\n"
                "本页由离线模式生成，仅作演示结构占位。请配置 LLM_API_KEY 后重建 Wiki。\n"
            )
        if "证据" in system or "citation" in system.lower() or "[" in user:
            if "临床" in system:
                return (
                    "**结论**：现有证据支持在规范生活方式干预的基础上，"
                    "对血压/血脂管理按指南使用药物治疗[1]。\n\n"
                    "**证据等级**：指南与荟萃分析为主（guideline/meta），"
                    "并含 RCT 支持。\n\n"
                    "**关键研究/指南**：[1] 高血压长期药物治疗的循证要点（guideline）。\n\n"
                    "**局限**：演示语料范围有限，结论需人工复核原文；"
                    "本系统不提供个体化处方与剂量建议。\n\n"
                    "参考文献见文末列表。"
                )
            if "营养" in system:
                return (
                    "**通俗结论**：均衡饮食与适度限盐对血压/血脂管理有帮助[1]。\n\n"
                    "**证据一句话**：现有研究提示地中海饮食、DASH 饮食等模式"
                    "与心血管风险降低相关。\n\n"
                    "**你可以怎么做**：多吃蔬果与全谷物，少盐少加工食品，"
                    "循序渐进地调整。\n\n"
                    "**何时就医**：已有慢性病或正在用药，调整饮食前先咨询医生。\n\n"
                    "参考文献见文末列表。"
                )
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


def with_json_mode_chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
) -> dict:
    """
    要求模型输出 JSON，并解析为 dict。

    参数:
        messages: 对话消息。
        temperature: 采样温度。

    返回:
        dict: 解析后的 JSON 对象。

    作用:
        供重排、大纲、评分解析等结构化任务复用，减少正则抠文本。
    """
    import re

    instructed = [
        *messages,
        {
            "role": "system",
            "content": (
                "只输出一个合法 JSON 对象，不要输出 Markdown 代码块或任何解释。"
            ),
        },
    ]
    raw = get_llm().chat(instructed, temperature=temperature, max_tokens=1200)
    cleaned = raw.strip()
    # 去掉可能的 ```json ... ``` 包裹
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.S)
    if fence:
        cleaned = fence.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 兜底：截取第一个 { 到最后一个 }
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"模型未返回合法 JSON: {raw[:200]!r}")


def embed_with_cache(texts: list[str], cache_dir: str | None = None) -> list[list[float]]:
    """
    带本地缓存的 embedding，避免重复计费。

    参数:
        texts: 文本列表。
        cache_dir: 缓存目录；None 时使用默认路径。

    返回:
        list[list[float]]: 与输入对齐的向量。

    作用:
        加快反复建库/调试时的向量化速度。
    """
    if not texts:
        return []
    cache_root = Path(cache_dir) if cache_dir else PROJECT_ROOT / "data" / "cache" / "embeddings"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / "embed_cache.json"

    cache: dict[str, list[float]] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}

    keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    missing: list[int] = []
    result: list[list[float]] = [cache.get(k) for k in keys]  # type: ignore[list-item]
    for i, v in enumerate(result):
        if v is None:
            missing.append(i)

    if missing:
        llm = get_llm()
        new_vecs = llm.embed([texts[i] for i in missing])
        for i, vec in zip(missing, new_vecs):
            cache[keys[i]] = vec
            result[i] = vec
        try:
            cache_file.write_text(
                json.dumps(cache, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as e:
            logger.warning("embed cache write failed: %s", e)
    return result  # type: ignore[return-value]
