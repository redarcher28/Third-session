# -*- coding: utf-8 -*-
"""
回答生成模块：带引用生成、拒答模板、Baseline（无检索）生成。
"""

from __future__ import annotations

import re
from typing import Any

from src.llm import get_llm
from src.models import Citation


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
        cites.append(
            Citation(
                index=i,
                doc_id=str(c.get("doc_id") or c.get("chunk_id") or f"ctx-{i}"),
                title=str(c.get("title") or ""),
                source=str(c.get("source") or ""),
                year=None if c.get("year") in (None, -1, "-1") else int(c["year"]),
                url=str(c.get("url") or ""),
                evidence_level=str(c.get("evidence_level") or "other"),
                snippet=str(c.get("text") or "")[:240],
            )
        )
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
        lines.append(
            f"[{c.index}] ({c.evidence_level}) {c.title} | {c.source} | {year} | {c.doc_id}\n"
            f"URL: {c.url}\n"
            f"{c.snippet}"
        )
    return "\n\n".join(lines)


def generate_answer(
    question: str,
    contexts: list[dict[str, Any]],
    *,
    system_persona: str,
    answer_style: str,
) -> tuple[str, list[Citation], bool]:
    """
    基于检索证据生成带引用回答；无证据则拒答。

    参数:
        question: 用户原问题。
        contexts: 检索证据 dict 列表。
        system_persona: 赛道人格系统提示。
        answer_style: 回答风格约束说明。

    返回:
        tuple:
            - answer (str): 回答正文（含声明）
            - citations (list[Citation]): 引用列表
            - refused (bool): 是否拒答
    """
    citations = contexts_to_citations(contexts)
    if not citations:
        return REFUSAL_TEMPLATE + DISCLAIMER, [], True

    context_block = format_context_block(citations)
    messages = [
        {
            "role": "system",
            "content": (
                f"{system_persona}\n\n"
                "硬性规则：\n"
                "1. 只能依据给定证据作答，禁止编造文献、PMID、NCT 或链接，不能用训练知识补全。\n"
                "2. 关键结论句末使用 [n] 引用编号，n 必须来自证据列表。\n"
                "3. 若证据不足以回答，明确说明证据不足，不要猜测。\n"
                "4. 若证据只能部分回答，先给出能确认的结论，再明确列出哪些方面缺乏证据支撑，不用推测补全。\n"
                "5. 证据来源少于 3 个时，用「研究提示」「可能」等弱化表述，不下确定结论。\n"
                "6. 证据冲突时并列呈现并说明不一致，不选边、不私自修正成唯一答案。\n"
                "7. 问题超出范围时明确说明边界并拒绝猜测。\n"
                "8. 文末用「参考文献」列出用到的 [n]。\n"
                f"9. 回答风格：{answer_style}\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题：{question}\n\n"
                f"证据列表：\n{context_block}\n\n"
                "请给出带引用的回答。"
            ),
        },
    ]
    llm = get_llm()
    answer = llm.chat(messages, temperature=0.2, max_tokens=1800)
    if DISCLAIMER.strip() not in answer:
        answer = answer.rstrip() + DISCLAIMER
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


def compute_faithfulness_proxy(
    answer: str,
    contexts: list[dict[str, Any]],
) -> float:
    """
    计算轻量忠实度代理分数（无需完整 Ragas 也可用）。

    参数:
        answer: 模型回答。
        contexts: 证据块列表。

    返回:
        float: 0~1，越高表示回答用词与证据重叠/支撑越好。

    作用:
        为评测增加可量化的「是否忠于检索内容」指标。
    """
    import re as _re

    if not answer.strip() or not contexts:
        return 0.0
    context_text = " ".join(
        str(c.get("text") or "") + " " + str(c.get("title") or "") for c in contexts
    ).lower()
    context_tokens = set(_re.findall(r"[\w\u4e00-\u9fff]+", context_text))
    # 去掉声明/参考文献等非正文部分，避免稀释分数
    body = _re.split(r"声明|参考文献|---", answer)[0]
    sentences = _re.split(r"[。！？!?；;\n]", body)
    scores: list[float] = []
    for sent in sentences:
        toks = _re.findall(r"[\w\u4e00-\u9fff]+", sent.lower())
        # 忽略过短或纯引用编号的片段
        if len(toks) < 3:
            continue
        hits = sum(1 for t in toks if t in context_tokens)
        scores.append(hits / len(toks))
    if not scores:
        return 0.0
    # 加权：长句权重更高
    weights = [len(_re.findall(r"[\w\u4e00-\u9fff]+", s)) for s in sentences if len(_re.findall(r"[\w\u4e00-\u9fff]+", s)) >= 3]
    total_w = sum(weights) or 1.0
    return round(sum(s * w for s, w in zip(scores, weights)) / total_w, 4)


def enforce_citation_density(
    answer: str,
    min_cites: int = 2,
) -> bool:
    """
    检查回答是否达到最低引用密度要求。

    参数:
        answer: 模型回答。
        min_cites: 至少应出现的不同 [n] 数量。

    返回:
        bool: True 表示满足最低引用要求。

    作用:
        防止「空口结论」；可在流水线中触发重生成。
    """
    return len(extract_citation_indices(answer)) >= min_cites


def format_reference_section(citations: list[Citation]) -> str:
    """
    按统一中文样式生成文末「参考文献」段落。

    参数:
        citations: Citation 列表。

    返回:
        str: 可直接拼接到回答末尾的参考文献 Markdown。

    作用:
        统一演示输出格式，便于评委核对。
    """
    if not citations:
        return ""
    lines = ["\n## 参考文献\n"]
    for c in citations:
        parts = [str(c.title or "未命名文献")]
        if c.source:
            parts.append(f"来源: {c.source}")
        if c.year:
            parts.append(f"年份: {c.year}")
        if c.evidence_level and c.evidence_level != "other":
            parts.append(f"证据等级: {c.evidence_level}")
        entry = " · ".join(parts)
        if c.url:
            entry += f"\n   {c.url}"
        lines.append(f"[{c.index}] {entry}")
    return "\n".join(lines)
