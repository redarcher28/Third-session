# -*- coding: utf-8 -*-
"""
混合检索：向量召回 + BM25 关键词 + 证据等级加权 + 可选 LLM 重排。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rank_bm25 import BM25Okapi

from src.kb.store import EvidenceStore
from src.llm import get_llm
from src.tracks.prompt_profiles import build_rerank_messages

logger = logging.getLogger(__name__)

# 证据等级加权系数（越大越优先）
LEVEL_WEIGHT = {
    "guideline": 1.25,
    "meta": 1.2,
    "rct": 1.15,
    "observational": 1.05,
    "wiki": 1.1,
    "ebook": 1.0,
    "other": 1.0,
}


def _tokenize(text: str) -> list[str]:
    """
    中英文分词（BM25 用）。

    创新点：中文连续字符按二元组（bigram）切分——中文无空格，
    整句会被切成一个词导致 BM25 完全失配；bigram 无需分词依赖即可工作。
    """
    tokens: list[str] = []
    for m in re.finditer(r"[\w\u4e00-\u9fff]+", text.lower()):
        seg = m.group()
        if re.fullmatch(r"[\u4e00-\u9fff]+", seg):
            if len(seg) == 1:
                tokens.append(seg)
            else:
                tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
        else:
            tokens.append(seg)
    return tokens


class HybridRetriever:
    """混合检索器。"""

    def __init__(self, store: EvidenceStore | None = None) -> None:
        """
        参数:
            store: 向量仓储；None 时新建默认 EvidenceStore。
        """
        self.store = store or EvidenceStore()
        self._bm25_docs: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._refresh_bm25()

    def _refresh_bm25(self) -> None:
        """从向量库导出语料并重建 BM25 索引。"""
        self._bm25_docs = self.store.all_chunks_for_bm25()
        if not self._bm25_docs:
            self._bm25 = None
            return
        corpus = [_tokenize(d.get("text", "")) for d in self._bm25_docs]
        corpus = [t if t else ["empty"] for t in corpus]
        self._bm25 = BM25Okapi(corpus)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_k: int = 16,
        prefer_levels: list[str] | None = None,
        boost_tags: list[str] | None = None,
        use_llm_rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """
        执行混合检索并返回最终证据列表。

        参数:
            query: 检索查询（建议已改写）。
            top_k: 最终返回条数。
            candidate_k: 召回候选条数。
            prefer_levels: 额外加权的证据等级列表。
            boost_tags: 额外加权的标签列表。
            use_llm_rerank: 是否用大模型重排。

        返回:
            list[dict]: 证据块字典列表（按相关度排序，已按 doc_id 去重）。
        """
        if self.store.count() == 0:
            return []

        # 没有远程 Embedding 时（离线哈希向量，或 Anthropic/AgentRouter
        # 单令牌配置）跳过 Chroma 向量分支，避免把无语义的伪向量当成召回依据；
        # 中文 bigram BM25 仍然负责关键词召回，RAG 链路不会断。
        llm = get_llm()
        # getattr 保留对旧版/测试替身 LLM（只有 is_offline 属性）的兼容。
        can_use_vectors = getattr(llm, "has_remote_embeddings", not llm.is_offline)
        vector_hits = [] if not can_use_vectors else self.store.query(query, n_results=candidate_k)
        bm25_hits = self._bm25_search(query, top_n=candidate_k)

        merged: dict[str, dict[str, Any]] = {}
        for rank, h in enumerate(vector_hits):
            cid = h["chunk_id"]
            score = 1.0 / (rank + 1)
            merged[cid] = {**h, "score": score, "from_vector": True, "from_bm25": False}
        for rank, h in enumerate(bm25_hits):
            cid = h["chunk_id"]
            score = 1.0 / (rank + 1)
            if cid in merged:
                merged[cid]["score"] += score
                merged[cid]["from_bm25"] = True
            else:
                merged[cid] = {**h, "score": score, "from_vector": False, "from_bm25": True}

        boost_tags = boost_tags or []
        prefer_levels = prefer_levels or []
        for item in merged.values():
            level = item.get("evidence_level", "other")
            item["score"] *= LEVEL_WEIGHT.get(str(level), 1.0)
            if prefer_levels and level in prefer_levels:
                item["score"] *= 1.15
            tags = str(item.get("tags") or "").split(",")
            if boost_tags and set(tags) & set(boost_tags):
                item["score"] *= 1.2
            # prefer wiki slightly for overview
            if str(item.get("source")) == "wiki":
                item["score"] *= 1.05

        candidates = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        candidates = candidates[: max(candidate_k, top_k)]

        if use_llm_rerank and len(candidates) > top_k:
            candidates = self._llm_rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        # Deduplicate by doc_id keeping best chunk
        seen_docs: set[str] = set()
        final: list[dict[str, Any]] = []
        for c in candidates:
            doc_id = str(c.get("doc_id") or c["chunk_id"])
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            final.append(c)
            if len(final) >= top_k:
                break
        return final

    def _bm25_search(self, query: str, top_n: int = 16) -> list[dict[str, Any]]:
        """关键词召回：返回 BM25 分数最高的 top_n 条。"""
        if not self._bm25 or not self._bm25_docs:
            self._refresh_bm25()
        if not self._bm25:
            return []
        tokens = _tokenize(query) or ["empty"]
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        return [{**self._bm25_docs[i], "bm25": float(s)} for i, s in ranked if s > 0]

    def _llm_rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """
        用大模型对候选证据重排。

        参数:
            query: 查询。
            candidates: 候选证据。
            top_k: 保留条数。

        返回:
            list[dict]: 重排后的前 top_k 条；失败则按原序截断。
        """
        llm = get_llm()
        messages = build_rerank_messages(query, candidates)
        try:
            raw = llm.chat(messages, temperature=0, max_tokens=80)
            nums = [int(x) for x in re.findall(r"\d+", raw)]
            ordered: list[dict[str, Any]] = []
            used: set[int] = set()
            for n in nums:
                if 1 <= n <= len(candidates) and n not in used:
                    ordered.append(candidates[n - 1])
                    used.add(n)
            for i, c in enumerate(candidates, start=1):
                if i not in used:
                    ordered.append(c)
            return ordered[:top_k]
        except Exception as e:
            logger.warning("LLM rerank failed: %s", e)
            return candidates[:top_k]


# ---------------------------------------------------------------------------
# 【待完善】增强检索型 RAG（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def score_evidence_priority(item: dict[str, Any]) -> float:
    """
    【待完善】按证据等级给单条结果打优先级分。

    参数:
        item: 检索结果 dict，至少含 evidence_level。

    返回:
        float: 权重分，越大越优先（建议指南>荟萃>RCT>观察>其他）。

    作用:
        供临床赛道加权排序，突出高质量证据。
    """
    raise NotImplementedError("待队员实现：score_evidence_priority")


def filter_by_year_range(
    items: list[dict[str, Any]],
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[dict[str, Any]]:
    """
    【待完善】按发表年份过滤检索结果。

    参数:
        items: 检索结果列表。
        year_from: 起始年（含），None 表示不限。
        year_to: 结束年（含），None 表示不限。

    返回:
        list[dict]: 过滤后的列表。

    作用:
        演示「近五年证据」等产品能力。
    """
    raise NotImplementedError("待队员实现：filter_by_year_range")


def explain_retrieval(query: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    """
    【待完善】生成检索过程的可解释说明（演示用）。

    参数:
        query: 用户或改写后的查询。
        items: 最终采用的证据列表。

    返回:
        dict: 例如 {
            "query": str,
            "why_selected": list[str],
            "sources": list[str],
            "notes": str,
        }

    作用:
        让评委看懂「为什么选了这些文献」，而不是黑盒。
    """
    raise NotImplementedError("待队员实现：explain_retrieval")


def diversify_by_source(
    items: list[dict[str, Any]],
    max_per_source: int = 2,
) -> list[dict[str, Any]]:
    """
    【待完善】按来源多样性重排，避免同一来源占满 Top-K。

    参数:
        items: 已排序的候选证据。
        max_per_source: 同一 source 最多保留条数。

    返回:
        list[dict]: 多样化后的列表。

    作用:
        增强检索型 RAG 的证据覆盖面（文献+试验+wiki 等）。
    """
    raise NotImplementedError("待队员实现：diversify_by_source")


def reciprocal_rank_fusion(
    vector_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    【待完善】对向量召回与 BM25 召回做 RRF 融合排序。

    参数:
        vector_hits: 向量检索结果（有序）。
        bm25_hits: BM25 检索结果（有序）。
        k: RRF 常数，常用 60。

    返回:
        list[dict]: 融合后的有序列表（含融合分）。

    作用:
        替代简单分值相加，提升混合检索稳定性。
    """
    raise NotImplementedError("待队员实现：reciprocal_rank_fusion")
