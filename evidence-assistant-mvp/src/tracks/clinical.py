# -*- coding: utf-8 -*-
"""
赛道一：临床证据助手 —— 人格、风格与查询改写。
"""

from __future__ import annotations

import json
import logging

from src.llm import get_llm, with_json_mode_chat
from src.retrieval.hybrid import score_evidence_priority

logger = logging.getLogger(__name__)

# 面向医生/医学生的系统人格
CLINICAL_PERSONA = (
    "你是面向临床医生与医学生的「临床证据助手」。"
    "回答应专业、结构化：结论 → 证据等级 → 关键研究/指南 → 局限。"
    "优先引用指南、荟萃分析与 RCT。"
)

# 回答风格约束（注入生成 Prompt）
CLINICAL_STYLE = (
    "使用专业术语；分点陈述；标明证据等级（guideline/meta/RCT/observational）；"
    "不做个体化处方与剂量建议。\n"
    "回答必须按「结论 → 证据等级 → 关键研究/指南 → 局限」四段组织，"
    "结论句前先汇总所依据的证据等级。"
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
                "将用户问题改写为适合检索医学文献的查询。\n"
                "输出格式（只输出一行，不要解释）：\n"
                "英文检索式（含疾病/干预/结局/研究类型关键词，可用 AND/OR 连接）"
                " || 中文核心关键词（逗号分隔）。\n"
                "示例：Hypertension AND antihypertensive therapy AND guideline"
                " || 高血压, 降压治疗, 指南\n"
                "如果输入包含「对话上下文」和「当前问题」，请结合上下文理解指代"
                "（如「它」「这个药」「那饮食呢」），只改写「当前问题」部分。"
            ),
        },
        {"role": "user", "content": question},
    ]
    return llm.chat(messages, temperature=0, max_tokens=180).strip() or question


# ---------------------------------------------------------------------------
# 临床赛道结构化增强
# ---------------------------------------------------------------------------


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
    # 有 LLM Key 时用 JSON 模式生成大纲，失败则回退到确定性规则
    try:
        llm = get_llm()
        if not llm.is_offline:
            data = with_json_mode_chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是临床证据助手。根据给定证据生成结构化回答大纲，"
                            "只输出 JSON，键为 conclusion、evidence_levels、"
                            "key_studies、limitations，均为字符串或字符串数组。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(contexts, ensure_ascii=False)[:4000],
                    },
                ]
            )
            if isinstance(data, dict):
                return {
                    "conclusion": str(data.get("conclusion", "")),
                    "evidence_levels": [
                        str(x) for x in data.get("evidence_levels", [])
                    ],
                    "key_studies": [str(x) for x in data.get("key_studies", [])],
                    "limitations": [str(x) for x in data.get("limitations", [])],
                }
    except Exception as e:
        logger.warning("clinical outline LLM failed, fallback: %s", e)

    ranked = rank_contexts_for_clinical(contexts)

    evidence_levels: list[str] = []
    for c in ranked:
        level = str(c.get("evidence_level") or "other")
        if level not in evidence_levels:
            evidence_levels.append(level)

    key_studies: list[str] = []
    for c in ranked:
        title = str(c.get("title") or "").strip()
        if not title:
            continue
        year = c.get("year")
        year_suffix = "" if year in (None, -1, "-1") else f"（{year}）"
        key_studies.append(f"【{str(c.get('evidence_level') or 'other')}】{title}{year_suffix}")
        if len(key_studies) >= 5:
            break

    if not ranked:
        conclusion = "当前检索未获得可用证据，建议补充公开文献后重建知识库。"
    else:
        top = ranked[0]
        conclusion = (
            f"基于当前证据，优先级最高的是"
            f"{str(top.get('evidence_level') or 'other')}类证据"
            f"「{str(top.get('title') or '检索到的证据')}」；"
            f"具体结论需结合患者个体情况并由医生复核。"
        )

    limitations = [
        "演示语料规模有限，未覆盖全部指南与研究；",
        "检索证据以摘要/切块为主，引用需人工复核原文；",
        "本系统不作个体化用药与剂量建议。",
    ]
    return {
        "conclusion": conclusion,
        "evidence_levels": evidence_levels,
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
    return sorted(contexts, key=score_evidence_priority, reverse=True)
