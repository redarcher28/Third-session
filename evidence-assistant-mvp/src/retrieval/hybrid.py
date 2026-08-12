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
from src.ingest import normalize_retrieved_context
from src.kb.weights import combined_priority
from src.llm import get_llm
from src.tracks.prompt_profiles import build_rerank_messages

logger = logging.getLogger(__name__)

# 试验注册：降权，避免与发表级 RCT 混淆
RECORD_TYPE_WEIGHT = {
    "trial_registry": 0.75,
    "wiki_page": 1.05,
    "published_article": 1.0,
}

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


def _is_citable_for_level_boost(item: dict[str, Any]) -> bool:
    """试验注册或未标记可引用的条目不享受 prefer_levels 加成。"""
    if not item.get("citation_eligible", True):
        return False
    if str(item.get("record_type") or "") == "trial_registry":
        return False
    return True


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

        # 没有远程 Embedding 时（离线哈希向量，或 Responses/Anthropic
        # 单令牌配置）跳过 Chroma 向量分支，避免把无语义的伪向量当成召回依据；
        # 中文 bigram BM25 仍然负责关键词召回，RAG 链路不会断。
        llm = get_llm()
        # getattr 保留对旧版/测试替身 LLM（只有 is_offline 属性）的兼容。
        can_use_vectors = getattr(llm, "has_remote_embeddings", not llm.is_offline)
        vector_hits: list[dict[str, Any]] = []
        if can_use_vectors:
            try:
                vector_hits = self.store.query(query, n_results=candidate_k)
            except Exception as exc:
                logger.warning("Vector retrieve failed (%s); BM25-only fallback", type(exc).__name__)
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
            record_type = str(item.get("record_type") or "other")
            item["score"] *= RECORD_TYPE_WEIGHT.get(record_type, 1.0)
            if prefer_levels and level in prefer_levels and _is_citable_for_level_boost(item):
                item["score"] *= 1.15
            tags = str(item.get("tags") or "").split(",")
            if boost_tags and set(tags) & set(boost_tags):
                item["score"] *= 1.2
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
        return [normalize_retrieved_context(c) for c in final]

    def retrieve_candidates(
        self,
        query: str,
        *,
        candidate_k: int = 20,
        prefer_levels: list[str] | None = None,
        boost_tags: list[str] | None = None,
        use_llm_rerank: bool = False,
    ) -> list[dict[str, Any]]:
        """
        返回扩大后的候选证据池，供充分性控制器筛选最小证据集。

        不改变 retrieve() 的既有 Top-K 语义；默认不做 LLM 重排，保证 BM25-only 可运行。
        """
        return self.retrieve(
            query,
            top_k=candidate_k,
            candidate_k=max(candidate_k, 16),
            prefer_levels=prefer_levels,
            boost_tags=boost_tags,
            use_llm_rerank=use_llm_rerank,
        )

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
    """按证据等级与年份给单条结果打优先级分（对接 A 组 weights）。"""
    level = str(item.get("evidence_level") or "other")
    year_raw = item.get("year")
    year: int | None
    if year_raw in (None, -1, "unknown", ""):
        year = None
    elif isinstance(year_raw, int):
        year = year_raw if year_raw > 0 else None
    else:
        try:
            year = int(str(year_raw))
        except (TypeError, ValueError):
            year = None
    return combined_priority(level, year)


def filter_by_year_range(
    items: list[dict[str, Any]],
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[dict[str, Any]]:
    """按发表年份过滤检索结果；未知年份在指定范围时保留。"""
    if year_from is None and year_to is None:
        return list(items)
    out: list[dict[str, Any]] = []
    for item in items:
        raw = item.get("year")
        if raw in (None, -1, "unknown", ""):
            out.append(item)
            continue
        try:
            year = int(raw)
        except (TypeError, ValueError):
            continue
        if year_from is not None and year < year_from:
            continue
        if year_to is not None and year > year_to:
            continue
        out.append(item)
    return out


def explain_retrieval(query: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    """生成检索过程的可解释说明（演示用）。"""
    why: list[str] = []
    sources: list[str] = []
    for i, item in enumerate(items[:5], start=1):
        title = str(item.get("title") or item.get("doc_id") or f"证据{i}")
        level = str(item.get("evidence_level") or "other")
        source = str(item.get("source") or "?")
        why.append(f"[{i}] {title} · {level} · {source}")
        if source not in sources:
            sources.append(source)
    return {
        "query": query,
        "why_selected": why,
        "sources": sources,
        "notes": f"共选用 {len(items)} 条证据片段。",
    }


def diversify_by_source(
    items: list[dict[str, Any]],
    max_per_source: int = 2,
) -> list[dict[str, Any]]:
    """按来源多样性重排，避免同一 source 占满 Top-K。"""
    if max_per_source <= 0:
        return list(items)
    counts: dict[str, int] = {}
    diversified: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for item in items:
        source = str(item.get("source") or "unknown")
        if counts.get(source, 0) < max_per_source:
            diversified.append(item)
            counts[source] = counts.get(source, 0) + 1
        else:
            deferred.append(item)
    return diversified + deferred


def reciprocal_rank_fusion(
    vector_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """对向量召回与 BM25 召回做 RRF 融合排序。"""
    scores: dict[str, float] = {}
    merged: dict[str, dict[str, Any]] = {}
    for rank, hit in enumerate(vector_hits):
        cid = str(hit.get("chunk_id") or hit.get("doc_id") or rank)
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        merged.setdefault(cid, hit)
    for rank, hit in enumerate(bm25_hits):
        cid = str(hit.get("chunk_id") or hit.get("doc_id") or f"bm25-{rank}")
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        merged.setdefault(cid, hit)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out: list[dict[str, Any]] = []
    for cid, rrf_score in ordered:
        row = dict(merged[cid])
        row["rrf_score"] = rrf_score
        out.append(row)
    return out
