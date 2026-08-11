# -*- coding: utf-8 -*-
"""
赛道一：临床证据助手 —— 人格、风格与查询改写。
"""

from __future__ import annotations

from typing import Any

from src.llm import get_llm
from src.retrieval.hybrid import score_evidence_priority

# 面向医生/医学生的系统人格
CLINICAL_PERSONA = (
    "你是面向临床医生与医学生的「临床证据助手」。"
    "回答应专业、结构化：结论 → 证据等级 → 关键研究/指南 → 局限。"
    "优先引用指南、荟萃分析与 RCT。"
)

# 回答风格约束（注入生成 Prompt）
CLINICAL_STYLE = (
    "使用专业术语；分点陈述；标明证据等级（guideline/meta/RCT/observational）；"
    "不做个体化处方与剂量建议。"
)

# 检索时优先的证据等级顺序相关列表
PREFER_LEVELS = ["guideline", "meta", "rct", "wiki", "observational"]


def rewrite_clinical_query(question: str) -> str:
    """
    将用户临床问题改写为更适合文献检索的查询。

    参数:
        question: 用户原问题（中文或中英混合）。

    返回:
        str: 改写后的检索查询；失败时回退为原问题。
    """
    llm = get_llm()
    messages = [
        {
            "role": "system",
            "content": (
                "将用户问题改写为适合检索医学文献的查询。"
                "输出格式（只输出一行，不要解释）："
                "英文检索式（含疾病/干预/结局/研究类型关键词，可用 AND/OR 连接）"
                " || 中文核心关键词（逗号分隔）。\n"
                "示例：Hypertension AND antihypertensive therapy AND guideline"
                " || 高血压, 降压治疗, 指南"
            ),
        },
        {"role": "user", "content": question},
    ]
    return llm.chat(messages, temperature=0, max_tokens=120).strip() or question


def build_clinical_answer_outline(contexts: list[dict]) -> dict:
    """
    根据证据生成临床回答大纲。

    参数:
        contexts: 检索证据列表。

    返回:
        dict: 建议包含
            - conclusion: str 结论要点
            - evidence_levels: list[str] 涉及证据等级
            - key_studies: list[str] 关键研究/指南标题
            - limitations: list[str] 局限

    作用:
        约束临床赛道输出结构，方便演示与人工打分。
    """
    levels: list[str] = []
    key_studies: list[str] = []
    limitations: list[str] = []

    ranked = rank_contexts_for_clinical(contexts)
    for item in ranked:
        level = str(item.get("evidence_level") or "other")
        if level not in levels:
            levels.append(level)
        title = str(item.get("title") or "").strip()
        if title and len(key_studies) < 3:
            year = item.get("year")
            year_s = f" ({year})" if year not in (None, -1, "-1") else ""
            key_studies.append(f"{title}{year_s} [{level}]")

    if not ranked:
        conclusion = "当前检索未返回可用证据，暂无法给出临床结论。"
    else:
        top = ranked[0]
        conclusion = (
            f"综合当前 {len(ranked)} 条证据，核心结论方向为："
            f"「{str(top.get('title') or '相关证据')}」所反映的临床要点。"
            "具体结论需结合证据正文与患者情况判断。"
        )
    if "guideline" not in levels:
        limitations.append("当前证据中缺少指南级资料，建议人工核对临床指南。")
    if "meta" not in levels and "rct" not in levels:
        limitations.append("缺少荟萃分析或 RCT，证据强度有限。")
    limitations.append("语料为窄领域演示集，不代表全科覆盖。")

    return {
        "conclusion": conclusion,
        "evidence_levels": levels,
        "key_studies": key_studies,
        "limitations": limitations,
    }


def rank_contexts_for_clinical(contexts: list[dict]) -> list[dict]:
    """
    按临床证据优先级对 contexts 重排。

    参数:
        contexts: 原始检索证据。

    返回:
        list[dict]: 重排后的证据列表。

    作用:
        确保指南/荟萃/RCT 更靠前展示与引用。
    """
    items: list[dict[str, Any]] = [
        dict(c) for c in contexts if isinstance(c, dict)
    ]
    items.sort(
        key=lambda x: (score_evidence_priority(x), float(x.get("score") or 0.0)),
        reverse=True,
    )
    return items
