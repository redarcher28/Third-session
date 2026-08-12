# -*- coding: utf-8 -*-
"""
赛道一/二统一编排流水线：改写 → 检索 →（可选在线补检索）→ 相关性检查 → 生成 → 引用校验。

对外主入口：ask(...)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from src.generation.answer import (
    REFUSAL_TEMPLATE,
    DISCLAIMER,
    format_reference_section,
    generate_answer,
)
from src.kb.chunking import docs_to_chunks
from src.llm import get_llm
from src.models import AskResponse, Citation
from src.retrieval.hybrid import (
    HybridRetriever,
    explain_retrieval,
    filter_by_year_range,
)
from src.tools.cite_check import (
    repair_answer_with_valid_cites,
    strip_invalid_claims,
    verify_citations,
)
from src.tools.live_search import search_clinical_trials, search_pubmed
from src.tracks.clinical import (
    CLINICAL_PERSONA,
    CLINICAL_STYLE,
    PREFER_LEVELS,
    build_clinical_answer_outline,
    rank_contexts_for_clinical,
    rewrite_clinical_query,
)
from src.tracks.nutrition import (
    BOOST_TAGS,
    NUTRITION_DOSAGE_REFUSAL,
    NUTRITION_PERSONA,
    NUTRITION_STYLE,
    build_nutrition_answer_outline,
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
    "statin", "blood pressure", "cardiovascular", "膳食纤维", "全谷物", "全谷",
    "含糖饮料", "甜饮料", "超加工", "植物性", "植物基", "低碳", "低碳水",
    "间歇性禁食", "轻断食", "肥胖", "减重", "体重", "坚果", "豆类", "鱼油",
    "omega", "whole grain", "dietary fiber", "sugar-sweetened", "ultra-processed",
    "plant-based", "low-carbohydrate", "intermittent fasting", "obesity",
}

# 明显越界/伪科学问题关键词：直接拒答（演示假引用与拒答能力）
OUT_OF_SCOPE = {
    "火星", "紫水晶", "水晶手链", "手链", "风水", "气功包治", "替代抗生素治肺炎",
    "外星人", "永动机", "mars dust", "crystal bracelet",
}


def _tokenize(text: str) -> set[str]:
    """提取中英文词元集合，供重叠度计算。"""
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def _is_low_relevance(
    question: str,
    contexts: list[dict[str, Any]],
    expect_levels: tuple[str, ...] | None = None,
) -> str | None:
    """
    判断是否应拒答，并返回拒答原因（None 表示可以正常回答）。

    材料第 29/31 页：
        - out_of_scope: 问题越界/伪科学；
        - low_relevance: 领域外或与证据几乎无重叠；
        - missing_evidence_type: 未命中所需证据类型（规则 2，临床赛道要求指南/RCT 级证据）。

    参数:
        question: 用户原问题。
        contexts: 检索到的证据。
        expect_levels: 期望的证据等级集合；命中任一即可放行。

    返回:
        str | None: 拒答原因，None 表示可正常回答。
    """
    q = question.lower()
    if any(term.lower() in q for term in OUT_OF_SCOPE):
        return "out_of_scope"
    if not any(h.lower() in q for h in DOMAIN_HINTS):
        q_tokens = _tokenize(question)
        if not contexts:
            return "low_relevance"
        overlap_scores = []
        for c in contexts[:3]:
            c_tokens = _tokenize(str(c.get("text") or "") + " " + str(c.get("title") or ""))
            if not q_tokens or not c_tokens:
                overlap_scores.append(0.0)
            else:
                overlap_scores.append(len(q_tokens & c_tokens) / max(1, len(q_tokens)))
        if max(overlap_scores, default=0.0) < 0.12:
            return "low_relevance"
    if not contexts:
        return "low_relevance"
    if expect_levels and not any(str(c.get("evidence_level")) in expect_levels for c in contexts):
        return "missing_evidence_type"
    return None


def _build_qualified_refusal(
    question: str,
    contexts: list[dict[str, Any]],
    reason: str,
) -> str:
    """
    合格拒答三要素：已检索到什么 + 缺什么 + 建议补查什么（材料第 30 页）。

    合格拒答不等于「我不能回答」——要说明检索结果与下一步，帮助用户/评委判断。
    """
    if not contexts:
        got = "未检索到任何证据片段"
    else:
        counts = Counter(str(c.get("source")) for c in contexts)
        got = f"检索到 {len(contexts)} 条证据，来源分布 {dict(counts)}"
    if reason == "out_of_scope":
        miss = "该问题不在本系统覆盖领域（高血压/血脂/糖尿病/心血管/饮食营养）内"
        suggest = "可补充该主题的公开指南或研究后重建知识库；必要时请咨询医疗机构"
    elif reason == "missing_evidence_type":
        miss = "检索到的证据等级不足（缺少指南/系统综述/RCT 级别证据）"
        suggest = "建议补充检索该主题的指南或随机对照试验（PubMed/ClinicalTrials）后重建知识库"
    else:
        miss = "证据相关性不足，无法支撑可靠结论"
        suggest = "建议补充检索该主题的指南或随机对照试验（PubMed/ClinicalTrials）后重建知识库"
    return (
        f"当前知识库中未找到能直接支持这一结论的高质量证据。\n"
        f"我能确认的是：{got}；但{miss}。\n"
        f"如需继续：{suggest}。\n"
        "本系统不提供个体化诊疗建议。"
    )


def _refusal_suggestions(question: str, reason: str) -> list[str]:
    """根据拒答原因生成改进问法的建议（演示用）。"""
    if reason == "out_of_scope":
        return [
            "把问题限定在本系统覆盖的医学领域（高血压、血脂、糖尿病、饮食营养等）",
            "换一个有公开证据的问题，例如「高血压 药物治疗 指南 证据」",
        ]
    if reason in ("off_domain", "low_relevance"):
        return [
            "补充更具体的疾病/干预关键词，例如「高血压 药物治疗 循证」",
            "尝试已知覆盖主题：血压、血脂、他汀、DASH/地中海饮食",
        ]
    if reason == "missing_evidence_type":
        return [
            "该主题当前缺少指南/荟萃/RCT 级证据，可尝试换一个更常见的临床问题",
            "或等待知识库补充该主题的指南与随机对照试验后重试",
        ]
    if reason == "empty_context":
        return [
            "检查是否开启了「近5年」或「仅高质量证据」等过严筛选",
            "可尝试开启「启用在线补检索」获取最新证据",
        ]
    return ["换个问法，加入更具体的疾病或干预词"]


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


def _retrieve_fused(
    retriever: HybridRetriever,
    queries: list[str],
    *,
    top_k: int,
    prefer_levels: list[str] | None,
    boost_tags: list[str] | None,
) -> list[dict[str, Any]]:
    """
    多查询检索并按 chunk 融合（RAG prompt 指南 Layer 1：查询改写/多路召回）。

    第一路用改写后的检索式（英文文献为主），其余路用用户原问题（中文关键词），
    两路结果按 chunk_id 融合去重后按分数排序；LLM 重排只对第一路执行以控制成本。
    """
    merged: dict[str, dict[str, Any]] = {}
    for i, q in enumerate(queries):
        if not q or not q.strip():
            continue
        items = retriever.retrieve(
            q,
            top_k=top_k,
            candidate_k=max(32, top_k * 4),
            prefer_levels=prefer_levels,
            boost_tags=boost_tags,
            use_llm_rerank=(i == 0),
        )
        for it in items:
            cid = str(it.get("chunk_id"))
            score = float(it.get("score") or 0.0)
            if cid in merged:
                merged[cid]["score"] = max(merged[cid]["score"], score)
            else:
                merged[cid] = it
    items = sorted(
        merged.values(),
        key=lambda x: float(x.get("score") or 0.0),
        reverse=True,
    )
    return items[:top_k]


def _retrieve_wiki_first(
    retriever: HybridRetriever,
    queries: list[str],
    *,
    top_k: int,
    prefer_levels: list[str] | None,
    boost_tags: list[str] | None,
) -> list[dict[str, Any]]:
    """
    Wiki 优先两级检索：先主题总览（source=wiki），再原文证据。

    与 _retrieve_fused 的区别：
        - 有真实向量服务时先走 select_wiki_then_chunks 拿主题页总览；
        - 再把多路融合结果并入去重，wiki 页排最前，原文按分数排序；
        - 向量服务不可用时直接退化为原多路融合检索（不引入哈希噪声）。
    """
    wiki_items: list[dict[str, Any]] = []
    if get_llm().embedding_available:
        try:
            from src.kb.wiki import select_wiki_then_chunks

            wiki_items = select_wiki_then_chunks(queries[0], wiki_k=1, chunk_k=top_k)
        except Exception as e:
            logger.warning("wiki-first retrieval failed, fallback to fused: %s", e)
            wiki_items = []
    fused = _retrieve_fused(
        retriever,
        queries,
        top_k=top_k,
        prefer_levels=prefer_levels,
        boost_tags=boost_tags,
    )
    merged: dict[str, dict[str, Any]] = {}
    for it in [*wiki_items, *fused]:
        cid = str(it.get("chunk_id") or it.get("doc_id") or "")
        if cid and cid not in merged:
            merged[cid] = it
    wiki = [it for it in merged.values() if it.get("kind") == "wiki"]
    evidence = [it for it in merged.values() if it.get("kind") != "wiki"]
    evidence.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return (wiki + evidence)[: max(top_k, 1)]


def ask(
    question: str,
    track: str = "clinical",
    *,
    top_k: int = 5,
    use_live_tools: bool = False,
    retriever: HybridRetriever | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    high_quality_only: bool = False,
) -> AskResponse:
    """
    赛道一/二统一问答入口。

    参数:
        question: 用户问题。
        track: "clinical" 或 "nutrition"。
        top_k: 最终证据条数。
        use_live_tools: 是否启用在线补检索。
        retriever: 可选外部注入的检索器（评测复用同一实例）。
        year_from: 发表年份下限（含），None 不限。
        year_to: 发表年份上限（含），None 不限。
        high_quality_only: 只保留指南/荟萃/RCT 证据（无高质量证据时回退）。

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
                citation_check={
                    "ok": True,
                    "has_citations": False,
                    "reason": "dosage_request",
                    "refusal_reason": "dosage_request",
                    "suggestions": _refusal_suggestions(question, "dosage_request"),
                },
            )
        rewritten = rewrite_nutrition_query(question)
        persona, style = NUTRITION_PERSONA, NUTRITION_STYLE
        prefer, boost = None, BOOST_TAGS
    else:
        track = "clinical"
        rewritten = rewrite_clinical_query(question)
        persona, style = CLINICAL_PERSONA, CLINICAL_STYLE
        prefer, boost = PREFER_LEVELS, None

    contexts = _retrieve_wiki_first(
        retriever,
        [rewritten, question],
        top_k=top_k,
        prefer_levels=prefer,
        boost_tags=boost,
    )
    if track == "clinical":
        contexts = rank_contexts_for_clinical(contexts)
    if year_from is not None or year_to is not None:
        contexts = filter_by_year_range(contexts, year_from=year_from, year_to=year_to)
    if high_quality_only:
        hq = [
            c
            for c in contexts
            if str(c.get("evidence_level")) in ("guideline", "meta", "rct")
        ]
        if hq:
            contexts = hq[:top_k]
    if use_live_tools:
        contexts = _live_augment(rewritten, contexts)[: top_k + 2]

    # 临床赛道要求指南/系统综述/RCT 级证据；wiki 主题页由高质量种子提炼，纳入门槛兜底
    expect_levels = tuple(PREFER_LEVELS[:4]) if track == "clinical" else None
    reject_reason = _is_low_relevance(question, contexts, expect_levels=expect_levels)
    if reject_reason:
        return AskResponse(
            answer=_build_qualified_refusal(question, contexts, reject_reason) + DISCLAIMER,
            citations=[],
            contexts=[],
            refused=True,
            rewritten_query=rewritten,
            track=track,
            citation_check={
                "ok": True,
                "has_citations": False,
                "reason": reject_reason,
                "refusal_reason": reject_reason,
                "suggestions": _refusal_suggestions(question, reject_reason),
            },
        )

    outline = None
    effective_style = style
    if track == "clinical":
        outline = build_clinical_answer_outline(contexts)
        effective_style = (
            style
            + "回答必须按「结论 → 证据等级 → 关键研究/指南 → 局限」四段组织，"
            "结论句前先汇总所依据的证据等级。"
        )
    elif track == "nutrition":
        outline = build_nutrition_answer_outline(contexts)
        effective_style = (
            style
            + "回答必须按「通俗结论 → 证据一句话 → 你可以怎么做 → 何时就医」"
            "四段组织，用词口语化、避免堆砌术语。"
        )

    answer, citations, refused = generate_answer(
        question,
        contexts,
        system_persona=persona,
        answer_style=effective_style,
    )
    if not refused and "参考文献" not in answer:
        answer = answer.rstrip() + format_reference_section(citations)
    check = verify_citations(answer, contexts)
    if not refused and not check.get("ok"):
        repaired = repair_answer_with_valid_cites(answer, contexts, check)
        if repaired != answer and verify_citations(repaired, contexts).get("ok"):
            answer = repaired
            check = verify_citations(answer, contexts)
    if outline is not None:
        if track == "clinical":
            check["clinical_outline"] = outline
        else:
            check["nutrition_outline"] = outline
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

    resp = AskResponse(
        answer=answer,
        citations=citations,
        contexts=ctx_citations,
        refused=refused,
        rewritten_query=rewritten,
        track=track,
        citation_check=check,
    )
    return attach_retrieval_explanation(resp, explain_retrieval(rewritten, contexts))


def detect_track_from_question(question: str) -> str:
    """
    根据问题内容自动判断应走 clinical 还是 nutrition。

    参数:
        question: 用户原问题。

    返回:
        str: "clinical" 或 "nutrition"。

    作用:
        支持演示「自动路由赛道」，减少用户手动选 Tab 的负担。
    """
    q = question.lower()
    # 饮食动作/食物类词权重更高：出现即强烈指向营养赛道
    nutrition_hints = {
        "饮食": 2, "吃": 2, "食物": 2, "膳食": 2, "食谱": 2, "减重": 2,
        "营养": 1, "盐": 1, "钠": 1, "地中海": 1, "维生素": 1,
        "diet": 2, "food": 2, "eat": 2, "nutrition": 1, "recipe": 2,
        "sodium": 1, "salt": 1, "mediterranean": 1, "dash": 1, "vitamin": 1,
        "meal": 1,
    }
    # 药物/治疗类词权重更高：出现即强烈指向临床赛道
    clinical_hints = {
        "药物": 2, "他汀": 2, "治疗": 2, "用药": 2, "剂量": 2,
        "血压": 1, "高血压": 1, "血脂": 1, "胆固醇": 1, "糖尿病": 1,
        "血糖": 1, "临床": 1, "患者": 1, "诊断": 1,
        "statin": 2, "drug": 2, "therapy": 2, "patient": 1, "clinical": 1,
        "hypertension": 1, "lipid": 1, "cholesterol": 1, "diabetes": 1,
    }
    n_score = sum(w for h, w in nutrition_hints.items() if h in q)
    c_score = sum(w for h, w in clinical_hints.items() if h in q)
    if n_score > c_score:
        return "nutrition"
    return "clinical"


def attach_retrieval_explanation(
    response: AskResponse,
    explanation: dict,
) -> AskResponse:
    """
    把检索可解释信息挂到 AskResponse（如写入 citation_check 扩展字段）。

    参数:
        response: 已有问答响应。
        explanation: explain_retrieval 的返回字典。

    返回:
        AskResponse: 附带解释信息后的响应。

    作用:
        打通检索解释 → 界面展示的数据通道。
    """
    check = dict(response.citation_check or {})
    check["retrieval_explanation"] = explanation
    return response.model_copy(update={"citation_check": check})
