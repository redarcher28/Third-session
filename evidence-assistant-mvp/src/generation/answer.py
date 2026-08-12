# -*- coding: utf-8 -*-
"""
回答生成模块：带引用生成、拒答模板、Baseline（无检索）生成。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from src.llm import get_llm
from src.models import Citation
from src.text_utils import context_to_citation_kwargs, filter_citable_contexts, truncate_at_sentence
from src.tracks.prompt_profiles import build_synthesis_messages


logger = logging.getLogger(__name__)


# 证据不足时的标准拒答文案
REFUSAL_TEMPLATE = (
    "当前知识库中未检索到足够相关、可核对的证据，无法给出有依据的回答。"
    "请尝试换一种问法，或补充公开文献后重建知识库。"
    "本系统不提供个体化诊疗建议。"
)

# 每次回答末尾追加的伦理声明
DISCLAIMER = (
    "\n\n---\n"
    "**声明**：本回答仅供学习与研究演示，不构成医疗建议，不能替代执业医师诊断与治疗。"
    "引用内容请人工复核原文。"
)

MODEL_UNAVAILABLE_TEMPLATE = (
    "当前模型服务暂时不可用，本次未生成综合回答。下面仅列出本次 RAG 实际召回的证据片段；"
    "请稍后重试，或先人工核对来源。"
)


def contexts_to_citations(contexts: list[dict[str, Any]]) -> list[Citation]:
    """
    将检索上下文转为带编号的 Citation 列表。

    参数:
        contexts: 检索返回的证据 dict 列表。

    返回:
        list[Citation]: index 从 1 开始。
    """
    cites: list[Citation] = []
    for i, c in enumerate(contexts, start=1):
        cites.append(Citation(**context_to_citation_kwargs(c, i)))
    return cites


def format_context_block(citations: list[Citation]) -> str:
    """
    把引用列表格式化为注入 Prompt 的证据文本块。

    参数:
        citations: Citation 列表。

    返回:
        str: 多段证据文本。
    """
    lines = []
    for c in citations:
        year = c.year or "n/a"
        body = truncate_at_sentence((c.text or c.snippet or ""), 800, min_keep=120)
        lines.append(
            f"[{c.index}] ({c.evidence_level}) {c.title} | {c.source} | {year} | {c.doc_id}\n"
            f"URL: {c.url}\n"
            f"{body}"
        )
    return "\n\n".join(lines)


def _evidence_only_fallback(citations: list[Citation]) -> str:
    """模型上游不可用时，返回明确标注的证据-only 结果。"""

    lines = [MODEL_UNAVAILABLE_TEMPLATE]
    for citation in citations:
        title = citation.title or citation.doc_id or "未命名来源"
        snippet = citation.snippet or "暂无可展示的证据片段"
        lines.append(f"[{citation.index}] {title}：{snippet}")
    return "\n\n".join(lines)


def _answer_overclaims(answer: str, allowed_claims: list[str]) -> bool:
    """粗检：回答是否引入与允许主张明显无关的疗效措辞。"""
    if not allowed_claims:
        return True
    body = answer
    for marker in (DISCLAIMER.strip(), "**参考文献**", "参考文献"):
        if marker and marker in body:
            body = body.split(marker)[0]
    # 若几乎不复述任何允许主张关键词，且出现强疗效句式，视为越界
    claim_blob = " ".join(allowed_claims)
    claim_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", claim_blob.lower()))
    body_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", body.lower()))
    overlap = len(claim_tokens & body_tokens) / max(1, len(claim_tokens))
    strong = any(
        k in body
        for k in ("一定能治愈", "保证降压", "可以停药", "替代药物治疗", "推荐剂量", "处方")
    )
    return strong or overlap < 0.08


def generate_answer(
    question: str,
    contexts: list[dict[str, Any]],
    *,
    system_persona: str,
    answer_style: str,
    track: str = "clinical",
    stream_callback: Callable[[str], None] | None = None,
    allowed_claims: list[str] | None = None,
    limitations: list[str] | None = None,
    missing_evidence: list[str] | None = None,
    evidence_status: str | None = None,
    force_evidence_card: bool = False,
) -> tuple[str, list[Citation], bool]:
    """
    基于检索证据生成带引用回答；无证据则拒答。

    参数:
        question: 用户原问题。
        contexts: 检索证据 dict 列表。
        system_persona: 赛道人格系统提示。
        answer_style: 回答风格约束说明。
        track: 赛道 key，用于加载统一 synthesis Prompt。
        allowed_claims / limitations / missing_evidence / evidence_status:
            主张卡充分性约束；存在时写入 synthesis Prompt。
        force_evidence_card: 跳过模型，直接输出证据卡模板。

    返回:
        tuple:
            - answer (str): 回答正文（含声明）
            - citations (list[Citation]): 引用列表
            - refused (bool): 是否拒答
    """
    citable = filter_citable_contexts(contexts)
    if not citable and evidence_status != "insufficient":
        return (
            "当前检索到的证据均不可作为可核验的外部引用（如本地无外链文档）。"
            "请在证据面板查看检索片段，或补充带公开链接的文献后重试。"
            + DISCLAIMER,
            [],
            True,
        )
    citations = contexts_to_citations(citable) if citable else contexts_to_citations(contexts)

    if evidence_status == "insufficient" or force_evidence_card:
        from src.retrieval.evidence_set import EvidencePlan, render_allowed_claims_answer

        plan = EvidencePlan(
            status=(evidence_status or "insufficient"),  # type: ignore[arg-type]
            allowed_claims=list(allowed_claims or []),
            limitations=list(limitations or []),
            missing_evidence=list(missing_evidence or []),
        )
        card = render_allowed_claims_answer(plan) + DISCLAIMER
        if stream_callback:
            stream_callback(card)
        refused = evidence_status == "insufficient" or (
            force_evidence_card and evidence_status not in {"partial", "sufficient"}
        )
        return card, citations, refused

    # 流式场景下越界难回滚：有主张约束时提前用确定性证据卡
    if (
        stream_callback is not None
        and evidence_status in {"partial", "sufficient"}
        and allowed_claims is not None
    ):
        from src.retrieval.evidence_set import EvidencePlan, render_allowed_claims_answer

        plan = EvidencePlan(
            status=evidence_status,  # type: ignore[arg-type]
            allowed_claims=list(allowed_claims),
            limitations=list(limitations or []),
            missing_evidence=list(missing_evidence or []),
        )
        card = render_allowed_claims_answer(plan) + DISCLAIMER
        stream_callback(card)
        return card, citations, False

    context_block = format_context_block(citations)
    messages = build_synthesis_messages(
        track,
        question,
        context_block,
        system_persona=system_persona,
        answer_style=answer_style,
        allowed_claims=allowed_claims,
        limitations=limitations,
        missing_evidence=missing_evidence,
        evidence_status=evidence_status,
    )
    llm = get_llm()
    streamed_chunks: list[str] = []
    try:
        if stream_callback is None:
            answer = llm.chat(messages, temperature=0.2, max_tokens=1800)
        else:
            try:
                for chunk in llm.stream_chat(messages, temperature=0.2, max_tokens=1800):
                    if not chunk:
                        continue
                    streamed_chunks.append(str(chunk))
                    stream_callback(str(chunk))
                answer = "".join(streamed_chunks).strip()
                if not answer:
                    raise RuntimeError("流式模型返回中没有可读文本")
            except Exception:
                # 某些 OpenAI-compatible 网关只支持非流式 Responses；保留完整
                # 回答能力作为兼容回退，真正拿到增量后才把异常交给外层降级。
                if streamed_chunks:
                    raise
                logger.info("LLM stream unavailable; falling back to non-streaming chat")
                answer = llm.chat(messages, temperature=0.2, max_tokens=1800)
                if stream_callback and answer:
                    stream_callback(answer)
    except Exception as exc:
        # 不把供应商响应正文写入日志，避免令牌/请求内容随异常外泄；证据仍然
        # 可以通过 Sources 面板展示，但必须明确告诉用户没有完成模型综合。
        logger.warning(
            "LLM synthesis unavailable; returning evidence-only fallback (%s)",
            type(exc).__name__,
        )
        fallback = _evidence_only_fallback(citations) + DISCLAIMER
        if stream_callback:
            suffix = fallback if not streamed_chunks else "\n\n" + fallback
            stream_callback(suffix)
        return fallback, citations, True

    if allowed_claims and _answer_overclaims(answer, allowed_claims):
        from src.retrieval.evidence_set import EvidencePlan, render_allowed_claims_answer

        plan = EvidencePlan(
            status=(evidence_status or "partial"),  # type: ignore[arg-type]
            allowed_claims=list(allowed_claims),
            limitations=list(limitations or []),
            missing_evidence=list(missing_evidence or []),
        )
        card = render_allowed_claims_answer(plan) + DISCLAIMER
        # 流式已输出时无法撤回；追加确定性纠偏卡
        if stream_callback and streamed_chunks:
            stream_callback("\n\n---\n" + card)
            answer = answer.rstrip() + "\n\n---\n" + card
        else:
            if stream_callback:
                stream_callback(card)
            answer = card

    if DISCLAIMER.strip() not in answer:
        answer = answer.rstrip() + DISCLAIMER
        if stream_callback:
            stream_callback(DISCLAIMER)
    return answer, citations, False


def generate_baseline_answer(question: str, *, system_persona: str) -> str:
    """
    纯通用大模型回答（无检索、无知识库），用于赛道三对比。

    参数:
        question: 用户问题。
        system_persona: 系统人格提示。

    返回:
        str: Baseline 回答文本。
    """
    messages = [
        {
            "role": "system",
            "content": (
                f"{system_persona}\n"
                "你是通用大模型。可以自由回答医学/健康问题，"
                "若引用文献请尽量真实；不确定时请说明。"
                "回答末尾声明：不构成医疗建议。"
            ),
        },
        {"role": "user", "content": question},
    ]
    llm = get_llm()
    answer = llm.chat(messages, temperature=0.3, max_tokens=1500)
    if "医疗建议" not in answer:
        answer = answer.rstrip() + DISCLAIMER
    return answer


def extract_citation_indices(answer: str) -> list[int]:
    """
    从回答中提取所有 [n] 引用编号。

    参数:
        answer: 回答文本。

    返回:
        list[int]: 去重排序后的编号列表。
    """
    return sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})


# ---------------------------------------------------------------------------
# 【待完善】生成质量增强（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def compute_faithfulness_proxy(
    answer: str,
    contexts: list[dict[str, Any]],
) -> float:
    """轻量忠实度代理：正文词元与证据重叠比例，并小幅奖励正文引用。"""
    from src.tools.cite_check import answer_body_for_citation_check

    body = answer_body_for_citation_check(answer)
    if not body.strip() or not contexts:
        return 0.0
    body_tokens = set(re.findall(r"[\w\u4e00-\u9fff]{2,}", body.lower()))
    if not body_tokens:
        return 0.0
    evidence_tokens: set[str] = set()
    for ctx in contexts[:8]:
        blob = f"{ctx.get('title') or ''} {ctx.get('text') or ''}".lower()
        evidence_tokens.update(re.findall(r"[\w\u4e00-\u9fff]{2,}", blob))
    if not evidence_tokens:
        return 0.0
    overlap = len(body_tokens & evidence_tokens) / len(body_tokens)
    cite_bonus = 0.12 if extract_citation_indices(body) else 0.0
    return round(min(1.0, overlap + cite_bonus), 3)


def enforce_citation_density(
    answer: str,
    min_cites: int = 1,
) -> bool:
    """检查正文是否至少出现 min_cites 个不同 [n] 引用。"""
    body = answer
    disclaimer = DISCLAIMER.strip()
    if disclaimer and disclaimer in body:
        body = body[: body.rfind(disclaimer)].rstrip()
    for marker in ("**参考文献**", "\n参考文献"):
        idx = body.find(marker)
        if idx >= 0:
            body = body[:idx].rstrip()
    return len(extract_citation_indices(body)) >= min_cites


def format_reference_section(
    citations: list[Citation],
    *,
    used_indices: list[int] | None = None,
) -> str:
    """按统一中文样式生成文末「参考文献」段落（默认仅列出正文实际使用的编号）。"""
    if not citations:
        return ""
    if used_indices is not None:
        allowed = set(used_indices)
        citations = [c for c in citations if c.index in allowed]
    if not citations:
        return ""
    lines = ["\n\n---\n**参考文献**"]
    for citation in citations:
        title = citation.title or citation.doc_id or "未命名来源"
        year = citation.year if citation.year not in (None, -1) else "年份未知"
        level = citation.evidence_level or "other"
        source = citation.source or "unknown"
        record_note = ""
        if citation.record_type == "trial_registry" or (
            level == "other" and "clinicaltrials" in source
        ):
            status = citation.trial_status.strip()
            record_note = "（试验注册，非发表疗效结果"
            if status:
                record_note += f"，状态：{status}"
            record_note += "）"
        lines.append(
            f"[{citation.index}] {title}{record_note} · {level} · {source} · {year}"
        )
        if citation.url:
            lines.append(f"    链接：{citation.url}")
        snippet = (citation.snippet or citation.text or "").strip()
        if snippet:
            preview = truncate_at_sentence(snippet, 280, min_keep=80)
            lines.append(f"    摘要：{preview}")
    return "\n".join(lines)


def append_reference_section_if_needed(
    answer: str,
    citations: list[Citation],
    *,
    used_indices: list[int] | None = None,
) -> str:
    """在回答末尾追加参考文献块（仅包含正文实际引用的条目）。"""
    if not citations or "参考文献" in answer:
        return answer
    section = format_reference_section(citations, used_indices=used_indices)
    if not section.strip():
        return answer
    disclaimer = DISCLAIMER.strip()
    if disclaimer and disclaimer in answer:
        idx = answer.rfind(disclaimer)
        if idx >= 0:
            return answer[:idx].rstrip() + section + "\n\n" + answer[idx:]
    return answer.rstrip() + section


def finalize_grounded_answer(
    answer: str,
    citations: list[Citation],
    check: dict[str, Any],
) -> tuple[str, list[Citation], dict[str, Any]]:
    """引用校验通过后，仅追加正文实际使用的参考文献。"""
    if not check.get("ok"):
        check = {**check, "reason": check.get("reason") or "citation_check_failed"}
        return answer, [], check
    valid_used = list(check.get("valid_used_brackets") or [])
    used_citations = [c for c in citations if c.index in set(valid_used)]
    if used_citations:
        answer = append_reference_section_if_needed(
            answer,
            citations,
            used_indices=valid_used,
        )
    elif citations:
        check = {**check, "ok": False, "reason": "missing_body_citations"}
        return answer, [], check
    return answer, used_citations, check
