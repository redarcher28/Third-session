# -*- coding: utf-8 -*-
"""
赛道一/二统一编排流水线：改写 → 检索 →（可选在线补检索）→ 相关性检查 → 生成 → 引用校验。

对外主入口：ask(...)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from collections import Counter
from time import perf_counter
from typing import Any

from src.config import get_settings
from src.generation.answer import REFUSAL_TEMPLATE, DISCLAIMER, generate_answer, finalize_grounded_answer
from src.kb.chunking import docs_to_chunks
from src.models import AskResponse, Citation
from src.text_utils import context_to_citation_kwargs
from src.retrieval.evidence_set import build_evidence_plan
from src.retrieval.hybrid import HybridRetriever
from src.text_utils import filter_citable_contexts
from src.tools.citation_policy import apply_citation_failure_policy
from src.tools.cite_check import verify_citations
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
    append_nutrition_query_aliases,
    detect_dosage_request,
    flag_dosage_in_answer,
    rewrite_nutrition_query,
    simplify_medical_terms,
)
from src.tracks.prompt_profiles import PROMPT_LAYERS, PROMPT_VERSION, get_track_profile

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


def _context_dict_to_citation(index: int, context: dict[str, Any]) -> Citation:
    return Citation(**context_to_citation_kwargs(context, index))


def _contexts_to_citations(contexts: list[dict[str, Any]]) -> list[Citation]:
    """把已检索上下文转换为前端可展示的证据卡片。

    拒答并不代表没有检索结果：当证据等级或相关性不足时，用户仍需要看到
    “系统查到了什么、为什么不够”的证据面板。此前拒答分支把 contexts 清空，
    导致备用页和 Open WebUI 都只能显示一段文字，丢失了旧前端的来源卡片。
    """
    return [_context_dict_to_citation(i, c) for i, c in enumerate(contexts, start=1)]


def _tokenize(text: str) -> set[str]:
    """提取中英文词元集合，供重叠度计算。"""
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def _qualifies_for_expected_levels(c: dict[str, Any], expect_levels: tuple[str, ...]) -> bool:
    """试验注册不能充当 guideline/meta/rct 级证据。"""
    if str(c.get("record_type") or "") == "trial_registry":
        return False
    if not c.get("citation_eligible", True):
        return False
    return str(c.get("evidence_level")) in expect_levels


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
    if expect_levels and not any(_qualifies_for_expected_levels(c, expect_levels) for c in contexts):
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


def _retrieval_summary(
    contexts: list[dict[str, Any]],
    *,
    rewritten_query: str,
    top_k: int,
    use_live_tools: bool,
    timings_ms: dict[str, float] | None = None,
    query_reformulation_mode: str = "llm",
    evidence_plan_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成供前端和报告使用的轻量检索摘要，不暴露完整内部候选。"""
    sources = Counter(str(c.get("source") or "unknown") for c in contexts)
    levels = Counter(str(c.get("evidence_level") or "other") for c in contexts)
    summary = {
        "rewritten_query": rewritten_query,
        "query_reformulation_mode": query_reformulation_mode,
        "requested_top_k": top_k,
        "retrieved_count": len(contexts),
        "sources": dict(sources),
        "evidence_levels": dict(levels),
        "live_tools": use_live_tools,
        "prompt_layers": list(PROMPT_LAYERS),
    }
    if timings_ms:
        summary["timings_ms"] = dict(timings_ms)
    if evidence_plan_fields:
        summary.update(evidence_plan_fields)
    try:
        from src.kb.health import kb_health_report

        health = kb_health_report()
        summary["kb_status"] = health.get("status", "unknown")
        summary["degraded_reasons"] = list(health.get("degraded_reasons") or [])
    except Exception as exc:
        logger.debug("kb health snapshot skipped for retrieval summary: %s", exc)
    return summary


def _finish_timings(timings_ms: dict[str, float], started: float) -> dict[str, float]:
    """补充总耗时并统一保留一位小数，方便前端直接展示。"""

    timings_ms["total_ms"] = round((perf_counter() - started) * 1000, 1)
    return timings_ms


def _lexical_query_rewrite(question: str, track: str) -> str:
    """用确定性词法扩展替代交互请求中的额外远程改写调用。

    这不是回答生成：它只保留用户原词，并补充知识库中已有的证据类型/主题锚点。
    需要更强的语义改写时仍可通过 RAG_USE_LLM_QUERY_REWRITE=true 使用预置 Prompt。
    """
    if track == "nutrition":
        return append_nutrition_query_aliases(question, question)
    return f"{question.strip()} guideline RCT systematic review meta-analysis"[:500]


def reformulate_query(question: str, track: str) -> tuple[str, str]:
    """与 ask()/ReAct 共用的查询改写：返回 (改写后查询, 模式标签)。"""
    settings = get_settings()
    if track == "nutrition":
        if settings.rag_use_llm_query_rewrite:
            return rewrite_nutrition_query(question), "llm"
        return _lexical_query_rewrite(question, track), "lexical"
    if settings.rag_use_llm_query_rewrite:
        return rewrite_clinical_query(question), "llm"
    return _lexical_query_rewrite(question, track), "lexical"


def build_retrieval_summary(
    contexts: list[dict[str, Any]],
    *,
    rewritten_query: str,
    top_k: int,
    use_live_tools: bool,
    query_reformulation_mode: str = "lexical",
    timings_ms: dict[str, float] | None = None,
    evidence_plan_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """供 API / 前端展示的统一检索摘要。"""
    return _retrieval_summary(
        contexts,
        rewritten_query=rewritten_query,
        top_k=top_k,
        use_live_tools=use_live_tools,
        timings_ms=timings_ms,
        query_reformulation_mode=query_reformulation_mode,
        evidence_plan_fields=evidence_plan_fields,
    )


def ask(
    question: str,
    track: str = "clinical",
    *,
    top_k: int = 5,
    use_live_tools: bool = False,
    retriever: HybridRetriever | None = None,
    stream_callback: Callable[[str], None] | None = None,
) -> AskResponse:
    """
    赛道一/二统一问答入口。

    参数:
        question: 用户问题。
        track: "clinical" 或 "nutrition"。
        top_k: 最终证据条数。
        use_live_tools: 是否启用在线补检索。
        retriever: 可选外部注入的检索器（评测复用同一实例）。
        stream_callback: 可选的模型文本增量回调，仅供 Open WebUI SSE 使用。

    返回:
        AskResponse: 含回答、引用、证据面板、拒答标记、校验结果。
    """
    started = perf_counter()
    timings_ms: dict[str, float] = {}

    if track not in {"clinical", "nutrition"}:
        raise ValueError(f"unsupported track: {track}")

    retriever = retriever or HybridRetriever()
    profile = get_track_profile(track)
    settings = get_settings()

    if track == "nutrition":
        # 产品边界：问具体药量/剂量的问题直接通俗拒答，不走检索生成
        if detect_dosage_request(question):
            timings_ms["safety_guard_ms"] = round((perf_counter() - started) * 1000, 1)
            _finish_timings(timings_ms, started)
            return AskResponse(
                answer=NUTRITION_DOSAGE_REFUSAL + DISCLAIMER,
                citations=[],
                contexts=[],
                refused=True,
                rewritten_query=question,
                track=track,
                prompt_version=PROMPT_VERSION,
                retrieval=_retrieval_summary(
                    [],
                    rewritten_query=question,
                    top_k=top_k,
                    use_live_tools=use_live_tools,
                    timings_ms=timings_ms,
                    query_reformulation_mode="guarded",
                ),
                citation_check={"ok": True, "has_citations": False, "reason": "dosage_request"},
                timings_ms=timings_ms,
            )
        rewrite_started = perf_counter()
        rewritten, rewrite_mode = reformulate_query(question, track)
        timings_ms["query_reformulation_ms"] = round(
            (perf_counter() - rewrite_started) * 1000, 1
        )
        persona, style = profile.persona, profile.style
        prefer, boost = list(profile.prefer_levels) or None, list(profile.boost_tags) or None
    else:
        rewrite_started = perf_counter()
        rewritten, rewrite_mode = reformulate_query(question, track)
        timings_ms["query_reformulation_ms"] = round(
            (perf_counter() - rewrite_started) * 1000, 1
        )
        persona, style = profile.persona, profile.style
        prefer, boost = list(profile.prefer_levels) or None, list(profile.boost_tags) or None

    retrieval_started = perf_counter()
    # 候选池大于最终证据集；充分性控制器再压缩到 top_k
    candidate_k = max(16, min(24, top_k * 4))

    def _retrieve_pool(query: str, k: int) -> list[dict[str, Any]]:
        if hasattr(retriever, "retrieve_candidates"):
            return retriever.retrieve_candidates(  # type: ignore[attr-defined]
                query,
                candidate_k=k,
                prefer_levels=prefer,
                boost_tags=boost,
                use_llm_rerank=False,
            )
        return retriever.retrieve(
            query,
            top_k=k,
            prefer_levels=prefer,
            boost_tags=boost,
            use_llm_rerank=False,
        )

    candidates = _retrieve_pool(rewritten, candidate_k)
    if use_live_tools:
        candidates = _live_augment(rewritten, candidates)[: candidate_k + 2]

    plan = build_evidence_plan(
        question,
        initial_candidates=candidates,
        retrieve_fn=_retrieve_pool,
        max_evidence=top_k,
        max_rounds=2,
    )
    plan_fields = plan.to_retrieval_fields()
    if plan.status == "unmapped":
        # 未进入主张卡充分性判定：保留既有 Top-K 语义
        contexts = retriever.retrieve(
            rewritten,
            top_k=top_k,
            prefer_levels=prefer,
            boost_tags=boost,
            use_llm_rerank=settings.rag_use_llm_rerank,
        )
        if use_live_tools:
            contexts = _live_augment(rewritten, contexts)[: top_k + 2]
    else:
        contexts = list(plan.selected_contexts)[:top_k]
        if not contexts:
            contexts = candidates[:top_k]
    timings_ms["retrieval_ms"] = round((perf_counter() - retrieval_started) * 1000, 1)

    # 临床赛道要求指南/系统综述/RCT 级证据（规则 2：缺预期证据类型→标注不足）
    relevance_started = perf_counter()
    expect_levels = tuple(PREFER_LEVELS[:3]) if track == "clinical" else None
    reject_reason = None
    if plan.status == "insufficient":
        reject_reason = "missing_evidence_type"
    elif plan.status == "unmapped":
        reject_reason = _is_low_relevance(question, contexts, expect_levels=expect_levels)
    timings_ms["relevance_check_ms"] = round((perf_counter() - relevance_started) * 1000, 1)
    if reject_reason and plan.status != "partial":
        _finish_timings(timings_ms, started)
        if plan.status == "insufficient":
            answer = (
                _build_qualified_refusal(question, contexts, reject_reason)
                + "\n"
                + "；".join(plan.missing_evidence)
                + DISCLAIMER
            )
        else:
            answer = _build_qualified_refusal(question, contexts, reject_reason) + DISCLAIMER
        return AskResponse(
            answer=answer,
            citations=[],
            # 证据不足时仍保留检索到的上下文，供旧证据栏和 Open WebUI 来源区展示。
            contexts=_contexts_to_citations(contexts),
            refused=True,
            rewritten_query=rewritten,
            track=track,
            prompt_version=PROMPT_VERSION,
            retrieval=_retrieval_summary(
                contexts,
                rewritten_query=rewritten,
                top_k=top_k,
                use_live_tools=use_live_tools,
                timings_ms=timings_ms,
                query_reformulation_mode=rewrite_mode,
                evidence_plan_fields=plan_fields,
            ),
            citation_check={"ok": True, "has_citations": False, "reason": reject_reason},
            timings_ms=timings_ms,
        )

    generation_started = perf_counter()
    gen_kwargs: dict[str, Any] = {}
    if plan.status in {"partial", "sufficient", "insufficient"}:
        gen_kwargs = {
            "allowed_claims": plan.allowed_claims,
            "limitations": plan.limitations,
            "missing_evidence": plan.missing_evidence,
            "evidence_status": plan.status,
        }
    answer, citations, refused = generate_answer(
        question,
        contexts,
        system_persona=persona,
        answer_style=style,
        track=track,
        stream_callback=stream_callback,
        **gen_kwargs,
    )
    timings_ms["generation_ms"] = round((perf_counter() - generation_started) * 1000, 1)
    validation_started = perf_counter()
    citable_contexts = filter_citable_contexts(contexts)
    check = verify_citations(answer, citable_contexts)
    if not refused:
        answer, citations, check = apply_citation_failure_policy(
            answer, contexts, citations, check, refused=refused
        )
        if check.get("ok"):
            answer, citations, check = finalize_grounded_answer(answer, citations, check)
        elif plan.status in {"partial", "sufficient"} and plan.allowed_claims:
            # 引用失败 → 确定性证据卡，避免误用「模型不可用」措辞掩盖策略降级
            from src.retrieval.evidence_set import render_allowed_claims_answer

            answer = render_allowed_claims_answer(plan) + DISCLAIMER
            citations = []
            check = {
                **check,
                "ok": False,
                "reason": check.get("reason") or "citation_check_failed_evidence_card",
            }
    # 营养赛道后处理：术语通俗化 + 药量防御警示（面向普通群众的产品边界）
    if not refused and track == "nutrition":
        # 术语替换需要看到完整回答；流式分支保留模型已经按赛道 Prompt
        # 生成的文本，避免事后替换造成前端已显示内容无法回退。
        if stream_callback is None:
            answer = simplify_medical_terms(answer)
        if flag_dosage_in_answer(answer):
            dosage_warning = (
                "\n\n> ⚠️ 如涉及具体药量/剂量，请以医生或药师的意见为准，切勿自行调整。"
            )
            if dosage_warning not in answer:
                answer = answer.rstrip() + dosage_warning
                if stream_callback:
                    stream_callback(dosage_warning)
    timings_ms["citation_validation_ms"] = round(
        (perf_counter() - validation_started) * 1000, 1
    )
    _finish_timings(timings_ms, started)

    ctx_citations = _contexts_to_citations(contexts)

    return AskResponse(
        answer=answer,
        citations=citations,
        contexts=ctx_citations,
        refused=refused,
        rewritten_query=rewritten,
        track=track,
        prompt_version=PROMPT_VERSION,
        retrieval=_retrieval_summary(
            contexts,
            rewritten_query=rewritten,
            top_k=top_k,
            use_live_tools=use_live_tools,
            timings_ms=timings_ms,
            query_reformulation_mode=rewrite_mode,
            evidence_plan_fields=plan_fields,
        ),
        citation_check=check,
        timings_ms=timings_ms,
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
