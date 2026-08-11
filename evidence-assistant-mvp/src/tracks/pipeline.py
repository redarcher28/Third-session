# -*- coding: utf-8 -*-
"""
赛道一/二统一编排流水线：改写 → 检索 →（可选在线补检索）→ 相关性检查 → 生成 → 引用校验。

对外主入口：ask(...)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.generation.answer import REFUSAL_TEMPLATE, DISCLAIMER, generate_answer
from src.kb.chunking import docs_to_chunks
from src.models import AskResponse, Citation
from src.retrieval.hybrid import HybridRetriever
from src.tools.cite_check import strip_invalid_claims, verify_citations
from src.tools.live_search import search_clinical_trials, search_pubmed
from src.tracks.clinical import (
    CLINICAL_PERSONA,
    CLINICAL_STYLE,
    PREFER_LEVELS,
    rewrite_clinical_query,
)
from src.tracks.nutrition import (
    BOOST_TAGS,
    NUTRITION_DOSAGE_REFUSAL,
    NUTRITION_PERSONA,
    NUTRITION_STYLE,
    detect_dosage_request,
    flag_dosage_in_answer,
    rewrite_nutrition_query,
    simplify_medical_terms,
)

logger = logging.getLogger(__name__)

# 领域关键词：用于粗判问题是否在本系统覆盖范围
DOMAIN_HINTS = {
    "高血压", "血压", "血脂", "胆固醇", "糖尿病", "血糖", "心血管", "地中海",
    "限钠", "钠", "盐", "饮食", "营养", "他汀", "dash", "hypertension",
    "lipid", "cholesterol", "diabetes", "diet", "mediterranean", "sodium",
    "statin", "blood pressure", "cardiovascular",
}

# 明显越界/伪科学问题关键词：直接拒答（演示假引用与拒答能力）
OUT_OF_SCOPE = {
    "火星", "紫水晶", "水晶手链", "手链", "风水", "气功包治", "替代抗生素治肺炎",
    "外星人", "永动机", "mars dust", "crystal bracelet",
}


def _tokenize(text: str) -> set[str]:
    """提取中英文词元集合，供重叠度计算。"""
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def _is_low_relevance(question: str, contexts: list[dict[str, Any]]) -> bool:
    """
    判断是否应拒答（越界、领域外、或与证据几乎无重叠）。

    参数:
        question: 用户原问题。
        contexts: 检索到的证据。

    返回:
        bool: True 表示应拒答。
    """
    q = question.lower()
    if any(term.lower() in q for term in OUT_OF_SCOPE):
        return True
    if not any(h.lower() in q for h in DOMAIN_HINTS):
        q_tokens = _tokenize(question)
        if not contexts:
            return True
        overlap_scores = []
        for c in contexts[:3]:
            c_tokens = _tokenize(str(c.get("text") or "") + " " + str(c.get("title") or ""))
            if not q_tokens or not c_tokens:
                overlap_scores.append(0.0)
            else:
                overlap_scores.append(len(q_tokens & c_tokens) / max(1, len(q_tokens)))
        return max(overlap_scores, default=0.0) < 0.12
    if not contexts:
        return True
    return False


def _live_augment(query: str, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    可选：在线补拉 PubMed/临床试验证据，插入到 contexts 前端。

    参数:
        query: 改写后查询。
        contexts: 已有离线检索结果。

    返回:
        list[dict]: 增强后的证据列表。
    """
    extra: list[dict[str, Any]] = []
    try:
        pubs = search_pubmed(query, retmax=3)
        for d in pubs:
            chunk = docs_to_chunks([d])[0]
            extra.append(chunk.model_dump())
    except Exception as e:
        logger.warning("live pubmed failed: %s", e)
    try:
        trials = search_clinical_trials(query.split()[0] if query else "hypertension", page_size=2)
        for d in trials:
            chunk = docs_to_chunks([d])[0]
            extra.append(chunk.model_dump())
    except Exception as e:
        logger.warning("live trials failed: %s", e)
    seen = {str(c.get("doc_id")) for c in contexts}
    for e in extra:
        if str(e.get("doc_id")) not in seen:
            contexts.insert(0, e)
            seen.add(str(e.get("doc_id")))
    return contexts


def ask(
    question: str,
    track: str = "clinical",
    *,
    top_k: int = 5,
    use_live_tools: bool = False,
    retriever: HybridRetriever | None = None,
) -> AskResponse:
    """
    赛道一/二统一问答入口。

    参数:
        question: 用户问题。
        track: "clinical" 或 "nutrition"。
        top_k: 最终证据条数。
        use_live_tools: 是否启用在线补检索。
        retriever: 可选外部注入的检索器（评测复用同一实例）。

    返回:
        AskResponse: 含回答、引用、证据面板、拒答标记、校验结果。
    """
    retriever = retriever or HybridRetriever()

    if track == "nutrition":
        # 产品边界：问具体药量/剂量的问题直接通俗拒答，不走检索生成
        if detect_dosage_request(question):
            return AskResponse(
                answer=NUTRITION_DOSAGE_REFUSAL + DISCLAIMER,
                citations=[],
                contexts=[],
                refused=True,
                rewritten_query=question,
                track=track,
                citation_check={"ok": True, "has_citations": False, "reason": "dosage_request"},
            )
        rewritten = rewrite_nutrition_query(question)
        persona, style = NUTRITION_PERSONA, NUTRITION_STYLE
        prefer, boost = None, BOOST_TAGS
    else:
        track = "clinical"
        rewritten = rewrite_clinical_query(question)
        persona, style = CLINICAL_PERSONA, CLINICAL_STYLE
        prefer, boost = PREFER_LEVELS, None

    contexts = retriever.retrieve(
        rewritten,
        top_k=top_k,
        prefer_levels=prefer,
        boost_tags=boost,
    )
    if use_live_tools:
        contexts = _live_augment(rewritten, contexts)[: top_k + 2]

    if _is_low_relevance(question, contexts):
        return AskResponse(
            answer=REFUSAL_TEMPLATE + DISCLAIMER,
            citations=[],
            contexts=[],
            refused=True,
            rewritten_query=rewritten,
            track=track,
            citation_check={"ok": True, "has_citations": False, "reason": "low_relevance"},
        )

    answer, citations, refused = generate_answer(
        question,
        contexts,
        system_persona=persona,
        answer_style=style,
    )
    check = verify_citations(answer, contexts)
    if not refused:
        answer = strip_invalid_claims(answer, check)
    # 营养赛道后处理：术语通俗化 + 药量防御警示（面向普通群众的产品边界）
    if not refused and track == "nutrition":
        answer = simplify_medical_terms(answer)
        if flag_dosage_in_answer(answer):
            answer = answer.rstrip() + (
                "\n\n> ⚠️ 如涉及具体药量/剂量，请以医生或药师的意见为准，切勿自行调整。"
            )

    ctx_citations = [
        Citation(
            index=i,
            doc_id=str(c.get("doc_id") or ""),
            title=str(c.get("title") or ""),
            source=str(c.get("source") or ""),
            year=None if c.get("year") in (None, -1, "-1") else int(c["year"]),
            url=str(c.get("url") or ""),
            evidence_level=str(c.get("evidence_level") or "other"),
            snippet=str(c.get("text") or "")[:240],
        )
        for i, c in enumerate(contexts, start=1)
    ]

    return AskResponse(
        answer=answer,
        citations=citations,
        contexts=ctx_citations,
        refused=refused,
        rewritten_query=rewritten,
        track=track,
        citation_check=check,
    )


# ---------------------------------------------------------------------------
# 【待完善】编排流水线增强（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def detect_track_from_question(question: str) -> str:
    """
    【待完善】根据问题内容自动判断应走 clinical 还是 nutrition。

    参数:
        question: 用户原问题。

    返回:
        str: "clinical" 或 "nutrition"。

    作用:
        支持演示「自动路由赛道」，减少用户手动选 Tab 的负担。
    """
    raise NotImplementedError("待队员实现：detect_track_from_question")


def attach_retrieval_explanation(
    response: AskResponse,
    explanation: dict,
) -> AskResponse:
    """
    【待完善】把检索可解释信息挂到 AskResponse（如写入 citation_check 扩展字段）。

    参数:
        response: 已有问答响应。
        explanation: explain_retrieval 的返回字典。

    返回:
        AskResponse: 附带解释信息后的响应。

    作用:
        打通检索解释 → 界面展示的数据通道。
    """
    raise NotImplementedError("待队员实现：attach_retrieval_explanation")
