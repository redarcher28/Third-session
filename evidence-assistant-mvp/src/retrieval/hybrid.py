# -*- coding: utf-8 -*-
"""
混合检索：向量召回 + BM25 关键词 + 证据等级加权 + 可选 LLM 重排。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rank_bm25 import BM25Okapi

from src.kb.weights import combined_priority
from src.kb.store import EvidenceStore
from src.llm import get_llm

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """中英文分词：中文附加字符二元组，改善中文关键词召回。"""
    tokens: list[str] = []
    for part in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.append(part)
            if len(part) > 1:
                tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
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
            list[dict]: 证据块字典列表（按相关度排序，已按 doc_id 去重并按来源多样化）。
        """
        if self.store.count() == 0:
            return []

        vector_hits = self.store.query(query, n_results=candidate_k)
        bm25_hits = self._bm25_search(query, top_n=candidate_k)

        # RRF 融合两路召回，替代简单分数相加
        merged = reciprocal_rank_fusion(vector_hits, bm25_hits, k=60)
        vector_ids = {h["chunk_id"] for h in vector_hits}
        bm25_ids = {h["chunk_id"] for h in bm25_hits}
        for item in merged:
            item["score"] = item.get("rrf_score", 0.0)
            item["from_vector"] = item["chunk_id"] in vector_ids
            item["from_bm25"] = item["chunk_id"] in bm25_ids

        boost_tags = boost_tags or []
        prefer_levels = prefer_levels or []
        for item in merged:
            item["score"] *= max(0.1, score_evidence_priority(item))
            level = str(item.get("evidence_level", "other"))
            if prefer_levels and level in prefer_levels:
                item["score"] *= 1.15
            tags = str(item.get("tags") or "").split(",")
            if boost_tags and set(tags) & set(boost_tags):
                item["score"] *= 1.2

        candidates = sorted(merged, key=lambda x: x["score"], reverse=True)
        candidates = candidates[: max(candidate_k, top_k)]

        if use_llm_rerank and len(candidates) > top_k:
            # 让重排输出完整候选顺序，后续再去重与多样化
            candidates = self._llm_rerank(query, candidates, top_k=len(candidates))
        else:
            candidates = candidates[: max(candidate_k, top_k)]

        # Deduplicate by doc_id keeping best chunk
        seen_docs: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for c in candidates:
            doc_id = str(c.get("doc_id") or c["chunk_id"])
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            deduped.append(c)

        # 来源多样性：先保证每来源都有名额，再用剩余候补按相关度补满
        max_per_source = max(1, top_k // 2)
        final = diversify_by_source(deduped, max_per_source=max_per_source)
        if len(final) < top_k:
            chosen_ids = {str(c.get("chunk_id") or c.get("doc_id")) for c in final}
            for c in deduped:
                if len(final) >= top_k:
                    break
                cid = str(c.get("chunk_id") or c.get("doc_id"))
                if cid not in chosen_ids:
                    final.append(c)
                    chosen_ids.add(cid)
        return final[:top_k]

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
        lines = []
        for i, c in enumerate(candidates, start=1):
            lines.append(
                f"[{i}] {c.get('title', '')}\n{(c.get('text') or '')[:280]}"
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是检索重排器。根据用户问题对候选证据按相关性排序，"
                    "只输出逗号分隔的编号，例如：3,1,5,2。不要解释。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{query}\n\n候选：\n" + "\n\n".join(lines),
            },
        ]
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
# 增强检索型 RAG 工具函数
# ---------------------------------------------------------------------------


def score_evidence_priority(item: dict[str, Any]) -> float:
    """
    按证据等级与发表年份给单条结果打优先级分（复用 A 组统一权重）。

    参数:
        item: 检索结果 dict，至少含 evidence_level。

    返回:
        float: 组合优先级分，越大越优先（等级越高、年份越新越高）。

    作用:
        供临床赛道加权排序，突出高质量证据。
    """
    level = str(item.get("evidence_level", "other"))
    year = item.get("year")
    if year in (None, -1, "", "-1"):
        year = None
    else:
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = None
    return combined_priority(level, year, level_w=1.0, recency_w=0.5)


def filter_by_year_range(
    items: list[dict[str, Any]],
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[dict[str, Any]]:
    """
    按发表年份过滤检索结果。

    参数:
        items: 检索结果列表。
        year_from: 起始年（含），None 表示不限。
        year_to: 结束年（含），None 表示不限。

    返回:
        list[dict]: 过滤后的列表。

    作用:
        演示「近五年证据」等产品能力。
    """
    out = []
    for item in items:
        year = item.get("year")
        if year in (None, -1, ""):
            # 无过滤条件时保留未知年份，有过滤条件时丢弃
            if year_from is None and year_to is None:
                out.append(item)
            continue
        year = int(year)
        if year_from is not None and year < year_from:
            continue
        if year_to is not None and year > year_to:
            continue
        out.append(item)
    return out


def explain_retrieval(query: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    """
    生成检索过程的可解释说明（演示用）。

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
    # 收集来源（去重）
    source_set = set()
    for item in items:
        source_set.add(item.get("source", "unknown"))

    # 为每条证据生成解释
    reasons = []
    for item in items:
        title = item.get("title", "未知文献")
        level = item.get("evidence_level", "other")
        reason = f"{level}类证据「{title}」"
        reasons.append(reason)

    # 生成总结
    count = len(items)
    source_count = len(source_set)
    notes = f"共检索到 {count} 条证据，来自 {source_count} 个来源"

    return {
        "query": query,
        "why_selected": reasons,
        "sources": sorted(source_set),
        "notes": notes,
    }



def diversify_by_source(
    items: list[dict[str, Any]],
    max_per_source: int = 2,
) -> list[dict[str, Any]]:
    """
    按来源多样性重排，避免同一来源占满 Top-K。

    参数:
        items: 已排序的候选证据。
        max_per_source: 同一 source 最多保留条数。

    返回:
        list[dict]: 多样化后的列表。

    作用:
        增强检索型 RAG 的证据覆盖面（文献+试验+wiki 等）。
    """
    count = {}
    new_list = []

    for item in items:
        source = item.get("source", "unknown")

        # 该来源当前已选几条（没出现过就是 0）
        current = count.get(source, 0)

        if current < max_per_source:
            new_list.append(item)
            count[source] = current + 1
    return new_list


def reciprocal_rank_fusion(
    vector_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    对向量召回与 BM25 召回做 RRF 融合排序。

    参数:
        vector_hits: 向量检索结果（有序）。
        bm25_hits: BM25 检索结果（有序）。
        k: RRF 常数，常用 60。

    返回:
        list[dict]: 融合后的有序列表（含融合分）。
    """
    scores = {}    # {chunk_id: RRF分数}
    doc_map = {}   # {chunk_id: 完整item}

    # 处理向量召回列表
    for rank, item in enumerate(vector_hits, start=1):
        cid = item["chunk_id"]
        scores[cid] = 1 / (k + rank)
        doc_map[cid] = item

    # 处理 BM25 召回列表（累加）
    for rank, item in enumerate(bm25_hits, start=1):
        cid = item["chunk_id"]
        rrf = 1 / (k + rank)
        if cid in scores:
            scores[cid] += rrf  # 两个列表都出现，累加
        else:
            scores[cid] = rrf  # 只在 BM25 出现
        doc_map[cid] = item    # 存完整信息

    # 按分数降序排列
    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    # 组装结果，把分数也带上
    result = []
    for cid in sorted_ids:
        item = doc_map[cid].copy()
        item["rrf_score"] = scores[cid]
        result.append(item)

    return result
