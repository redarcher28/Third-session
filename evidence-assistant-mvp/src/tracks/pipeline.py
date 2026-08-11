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
    DISCLAIMER,
    compute_faithfulness_proxy,
    enforce_citation_density,
    generate_answer,
)
from src.kb.chunking import docs_to_chunks
from src.kb.wiki import select_wiki_then_chunks
from src.llm import get_llm
from src.models import AskResponse, Citation, EvidenceDoc
from src.retrieval.hybrid import (
    HybridRetriever,
    explain_retrieval,
    filter_by_year_range,
)
from src.tools.cite_check import (
    detect_unsupported_claims,
    repair_answer_with_valid_cites,
    strip_invalid_claims,
    verify_citations,
)
from src.tools.live_search import (
    merge_live_and_offline_docs,
    search_clinical_trials,
    search_pubmed,
    should_trigger_live_search,
)
from src.tracks.clinical import (
    CLINICAL_PERSONA,
    CLINICAL_STYLE,
    PREFER_LEVELS,
    rank_contexts_for_clinical,
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


def contextualize_question(
    question: str,
    history: list[dict[str, str]] | None = None,
    *,
    max_turns: int = 4,
) -> str:
    """
    把最近几轮对话与当前问题拼成带上下文的文本，供改写 / 路由 / 拒答判断复用。

    参数:
        question: 用户当前问题。
        history: 历史消息列表（role: user/assistant，content: 文本）。
        max_turns: 最多带入几轮（一“轮”包含用户 + 助手两条消息）。

    返回:
        str: 带上下文的查询文本；无历史时原样返回问题。
    """
    if not history:
        return question
    lines: list[str] = []
    for m in history[-(max_turns * 2) :]:
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        label = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"{label}：{content[:600]}")
    if not lines:
        return question
    return "对话上下文：\n" + "\n".join(lines) + f"\n\n当前问题：{question}"


def _split_rewritten_query(rewritten: str) -> tuple[str, str]:
    """
    把「英文检索式 || 中文关键词」拆成两路查询；无分隔符时整体作为两路共用。

    参数:
        rewritten: 查询改写结果。

    返回:
        tuple[str, str]: (第一路英文检索式, 第二路中文关键词)。
    """
    text = rewritten.strip()
    if "||" in text:
        left, _, right = text.partition("||")
        return left.strip(), right.strip()
    return text, text


def _retrieve_fused(
    retriever: HybridRetriever,
    queries: list[str],
    *,
    top_k: int = 5,
    prefer_levels: tuple[str, ...] | None = None,
    boost_tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    双查询融合召回：改写式 + 原问题各跑一路，按 chunk_id 去重合并。

    参数:
        retriever: 混合检索器。
        queries: 多路查询（第一路走 LLM 重排，其余路关闭以控制成本）。
        top_k: 目标证据条数。
        prefer_levels: 临床赛道优先证据等级。
        boost_tags: 营养赛道加权标签。

    返回:
        list[dict]: 融合去重后的证据块列表。
    """
    fused: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, query in enumerate(queries):
        if not query.strip():
            continue
        hits = retriever.retrieve(
            query,
            top_k=max(top_k + 3, 6),
            prefer_levels=list(prefer_levels) if prefer_levels else None,
            boost_tags=boost_tags,
            use_llm_rerank=(i == 0),
        )
        for c in hits:
            cid = str(c.get("chunk_id") or c.get("doc_id"))
            if cid in seen:
                continue
            seen.add(cid)
            fused.append(c)
        if len(fused) >= top_k + 8:
            break
    return fused[: top_k + 8]


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


def _live_augment(
    query: str,
    contexts: list[dict[str, Any]],
    *,
    max_total: int = 8,
) -> list[dict[str, Any]]:
    """
    可选：在线补拉 PubMed/临床试验证据，插入到 contexts 前端。

    参数:
        query: 改写后查询。
        contexts: 已有离线检索结果。

    返回:
        list[dict]: 增强后的证据列表。
    """
    live_docs: list[EvidenceDoc] = []
    try:
        live_docs.extend(search_pubmed(query, retmax=3))
    except Exception as e:
        logger.warning("live pubmed failed: %s", e)
    try:
        live_docs.extend(
            search_clinical_trials(
                query.split()[0] if query else "hypertension", page_size=2
            )
        )
    except Exception as e:
        logger.warning("live trials failed: %s", e)
    if not live_docs:
        return contexts

    valid_sources = {"pubmed", "clinicaltrials", "europepmc", "local", "wiki"}
    offline_docs: list[EvidenceDoc] = []
    for c in contexts:
        raw_source = str(c.get("source") or "local")
        offline_docs.append(
            EvidenceDoc(
                doc_id=str(c.get("doc_id") or c.get("chunk_id") or ""),
                source=raw_source if raw_source in valid_sources else "local",
                title=str(c.get("title") or ""),
                text=str(c.get("text") or ""),
                year=None if c.get("year") in (None, -1, "-1") else int(c["year"]),
                url=str(c.get("url") or ""),
                tags=[],
                evidence_level=str(c.get("evidence_level") or "other"),
            )
        )

    merged_docs = merge_live_and_offline_docs(
        live_docs, offline_docs, max_total=max_total
    )
    out: list[dict[str, Any]] = []
    for d in merged_docs:
        out.append(docs_to_chunks([d])[0].model_dump())
    return out


def ask(
    question: str,
    track: str = "clinical",
    *,
    top_k: int = 5,
    use_live_tools: bool = False,
    retriever: HybridRetriever | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    history: list[dict[str, str]] | None = None,
) -> AskResponse:
    """
    赛道一/二统一问答入口。

    参数:
        question: 用户问题。
        track: "clinical" 或 "nutrition"。
        top_k: 最终证据条数。
        use_live_tools: 是否启用在线补检索。
        retriever: 可选外部注入的检索器（评测复用同一实例）。
        year_from: 可选证据起始年份（含）。
        year_to: 可选证据结束年份（含）。
        history: 可选对话历史（role/content 字典列表），用于支持追问与指代消解。

    返回:
        AskResponse: 含回答、引用、证据面板、拒答标记、校验结果。
    """
    retriever = retriever or HybridRetriever()
    contextual = contextualize_question(question, history)

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
        rewritten = rewrite_nutrition_query(contextual)
        persona, style = NUTRITION_PERSONA, NUTRITION_STYLE
        prefer, boost = None, BOOST_TAGS
    else:
        track = "clinical"
        rewritten = rewrite_clinical_query(contextual)
        persona, style = CLINICAL_PERSONA, CLINICAL_STYLE
        prefer, boost = PREFER_LEVELS, None

    rewritten_en, rewritten_zh = _split_rewritten_query(rewritten)
    contexts = _retrieve_fused(
        retriever,
        [rewritten_en, contextual],
        top_k=top_k,
        prefer_levels=prefer,
        boost_tags=boost,
    )
    if use_live_tools or (
        not get_llm().is_offline and should_trigger_live_search(question, len(contexts))
    ):
        contexts = _live_augment(
            rewritten_en or rewritten, contexts, max_total=top_k + 2
        )

    # Wiki 总览优先：先给主题知识页，再补原文证据（A 组 select_wiki_then_chunks）
    try:
        wiki_first = select_wiki_then_chunks(
            rewritten_zh or contextual, wiki_k=min(2, top_k), chunk_k=top_k
        )
    except Exception as e:
        logger.warning("wiki-first retrieval failed: %s", e)
        wiki_first = []
    wiki_items = [h for h in wiki_first if h.get("kind") == "wiki"]
    if wiki_items:
        wiki_doc_ids = {
            str(h.get("doc_id") or h.get("chunk_id")) for h in wiki_items
        }
        rest = [
            c
            for c in contexts
            if str(c.get("doc_id") or c.get("chunk_id")) not in wiki_doc_ids
        ]
        if track == "clinical":
            rest = rank_contexts_for_clinical(rest)
        contexts = [*wiki_items, *rest][: top_k + 4]
    elif track == "clinical":
        contexts = rank_contexts_for_clinical(contexts)
    for c in contexts:
        c.setdefault("kind", "wiki" if str(c.get("source")) == "wiki" else "evidence")

    if year_from is not None or year_to is not None:
        contexts = filter_by_year_range(
            contexts, year_from=year_from, year_to=year_to
        )

    # 临床赛道要求指南/系统综述/RCT 级证据（规则 2：缺预期证据类型→标注不足）
    expect_levels = tuple(PREFER_LEVELS[:3]) if track == "clinical" else None
    reject_reason = _is_low_relevance(contextual, contexts, expect_levels=expect_levels)
    if reject_reason:
        return AskResponse(
            answer=_build_qualified_refusal(question, contexts, reject_reason) + DISCLAIMER,
            citations=[],
            contexts=[],
            refused=True,
            rewritten_query=rewritten,
            track=track,
            citation_check={"ok": True, "has_citations": False, "reason": reject_reason},
        )

    answer, citations, refused = generate_answer(
        question,
        contexts,
        system_persona=persona,
        answer_style=style,
        history=history,
    )
    check = verify_citations(answer, contexts)
    if not refused:
        repaired = repair_answer_with_valid_cites(answer, contexts, check)
        if repaired != answer:
            answer = repaired
            check["repaired_citations"] = True
        else:
            answer = strip_invalid_claims(answer, check)
        check["citation_density_ok"] = enforce_citation_density(answer, min_cites=1)
        check["unsupported_claims"] = detect_unsupported_claims(answer, contexts)
        check["faithfulness"] = compute_faithfulness_proxy(answer, contexts)
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

    response = AskResponse(
        answer=answer,
        citations=citations,
        contexts=ctx_citations,
        refused=refused,
        rewritten_query=rewritten,
        track=track,
        citation_check=check,
    )
    if contexts:
        explanation = explain_retrieval(rewritten, contexts)
        response = attach_retrieval_explanation(response, explanation)
    return response


# ---------------------------------------------------------------------------
# 编排流水线增强
# ---------------------------------------------------------------------------


def detect_track_from_question(
    question: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """
    根据问题内容自动判断应走 clinical 还是 nutrition。

    参数:
        question: 用户原问题。
        history: 可选对话历史；追问较短时结合前文判断赛道。

    返回:
        str: "clinical" 或 "nutrition"。

    作用:
        支持演示「自动路由赛道」，减少用户手动选 Tab 的负担。
    """
    text = contextualize_question(question, history).lower()
    nutrition_hints = (
        "饮食", "营养", "吃", "盐", "钠", "地中海", "dash", "膳食",
        "食物", "补剂", "维生素", "减重", "减肥", "零食",
    )
    clinical_hints = (
        "高血压", "血压", "血脂", "胆固醇", "糖尿病", "血糖", "心血管",
        "他汀", "药物", "治疗", "指南", "临床试验", "患者", "疾病",
        "症状", "剂量", "合并症",
    )
    if any(h in text for h in nutrition_hints):
        return "nutrition"
    if any(h in text for h in clinical_hints):
        return "clinical"
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
    citation_check = dict(response.citation_check)
    citation_check["explanation"] = explanation
    return response.model_copy(update={"citation_check": citation_check})
