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
    """简单中英文分词（整词），供关键词展示使用。"""
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def _bm25_tokenize(text: str) -> list[str]:
    """
    BM25 用分词：英文按词；中文整词 + 相邻二元组。

    中文整句作为单一词元时几乎无法匹配，加入二元组可显著提升中文检索召回。
    """
    tokens: list[str] = []
    for part in re.findall(r"[\w\u4e00-\u9fff]+", text.lower()):
        if re.search(r"[\u4e00-\u9fff]", part):
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
        corpus = [_bm25_tokenize(d.get("text", "")) for d in self._bm25_docs]
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

        # 未配置真实向量服务时，embedding 为哈希占位（无语义相似度），
        # 向量分支是纯噪声：跳过它让 BM25 关键词召回独挑大梁，保证检索质量。
        vector_hits = [] if not get_llm().embedding_available else self.store.query(query, n_results=candidate_k)
        bm25_hits = self._bm25_search(query, top_n=candidate_k)
        extra_lists: list[list[dict[str, Any]]] = []
        if prefer_levels:
            level_set = {str(x) for x in prefer_levels}
            level_docs = [
                d
                for d in self._bm25_docs
                if str(d.get("evidence_level")) in level_set
            ]
            if level_docs:
                extra_lists.append(
                    self._bm25_search(query, top_n=candidate_k, docs=level_docs)
                )

        # RRF 融合：排名倒数融合，避免单一召回头部分数淹没另一路的相关命中
        fused = reciprocal_rank_fusion(vector_hits, bm25_hits, *extra_lists, k=60)
        merged: dict[str, dict[str, Any]] = {}
        for item in fused:
            cid = item["chunk_id"]
            item["score"] = item.get("fusion_score", 0.0)
            merged[cid] = item

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

    def _bm25_search(
        self,
        query: str,
        top_n: int = 16,
        docs: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        关键词召回：返回 BM25 分数最高的 top_n 条。

        参数:
            query: 查询。
            top_n: 返回条数。
            docs: 可选语料子集（如仅高质量证据）；None 使用全量语料。
        """
        corpus_docs = docs if docs is not None else self._bm25_docs
        if not corpus_docs:
            return []
        corpus = [_bm25_tokenize(d.get("text", "")) for d in corpus_docs]
        corpus = [t if t else ["empty"] for t in corpus]
        index = BM25Okapi(corpus)
        tokens = _bm25_tokenize(query) or ["empty"]
        scores = index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        return [{**corpus_docs[i], "bm25": float(s)} for i, s in ranked if s > 0]

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


def score_evidence_priority(item: dict[str, Any]) -> float:
    """
    按证据等级给单条结果打优先级分。

    参数:
        item: 检索结果 dict，至少含 evidence_level。

    返回:
        float: 权重分，越大越优先（建议指南>荟萃>RCT>观察>其他）。

    作用:
        供临床赛道加权排序，突出高质量证据。
    """
    level = str(item.get("evidence_level") or "other").lower()
    return float(LEVEL_WEIGHT.get(level, 1.0))


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
    out: list[dict[str, Any]] = []
    for item in items:
        raw_year = item.get("year")
        try:
            year = int(raw_year)
        except (TypeError, ValueError):
            # 年份未知：仅在完全没有年份约束时保留
            if year_from is None and year_to is None:
                out.append(item)
            continue
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
    query_tokens = set(_tokenize(query))
    why_selected: list[str] = []
    sources: list[str] = []
    for i, item in enumerate(items, start=1):
        title = str(item.get("title") or "无标题")
        source = str(item.get("source") or "unknown")
        level = str(item.get("evidence_level") or "other")
        text = f"{item.get('text') or ''} {title}"
        hit_terms = sorted(query_tokens & set(_tokenize(text)))[:6]
        year = item.get("year")
        year_s = str(year) if year not in (None, -1, "-1") else "年份未知"
        if hit_terms:
            why = f"[{i}] {title}（{source}/{level}/{year_s}）：命中关键词「{'、'.join(hit_terms)}」"
        else:
            why = f"[{i}] {title}（{source}/{level}/{year_s}）：语义相近或来源加权"
        why_selected.append(why)
        if source not in sources:
            sources.append(source)
    notes = (
        f"共召回 {len(items)} 条证据，覆盖来源：{'、'.join(sources) or '无'}。"
        "排序综合了向量相关度、BM25 关键词与证据等级加权。"
    )
    return {
        "query": query,
        "why_selected": why_selected,
        "sources": sources,
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
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for item in items:
        src = str(item.get("source") or "unknown")
        if src not in groups:
            groups[src] = []
            order.append(src)
        groups[src].append(item)
    result: list[dict[str, Any]] = []
    idx = {src: 0 for src in order}
    active = True
    while active:
        active = False
        for src in order:
            if idx[src] < len(groups[src]) and idx[src] < max_per_source:
                result.append(groups[src][idx[src]])
                idx[src] += 1
                active = True
    return result


def reciprocal_rank_fusion(
    *ranked_lists: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    对多路召回做 RRF 融合排序（向量 / BM25 / 等级限定 BM25 等）。

    参数:
        *ranked_lists: 多路有序召回结果。
        k: RRF 常数，常用 60。

    返回:
        list[dict]: 融合后的有序列表（含融合分）。

    作用:
        替代简单分值相加，提升混合检索稳定性。
    """
    merged: dict[str, dict[str, Any]] = {}
    for lst in ranked_lists:
        for rank, hit in enumerate(lst, start=1):
            cid = str(hit.get("chunk_id") or f"r-{rank}")
            if cid in merged:
                merged[cid]["fusion_score"] += 1.0 / (k + rank)
            else:
                merged[cid] = {**hit, "fusion_score": 1.0 / (k + rank)}
    fused = sorted(merged.values(), key=lambda x: x["fusion_score"], reverse=True)
    return fused
