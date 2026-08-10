# -*- coding: utf-8 -*-
"""
赛道一：临床证据助手 —— 人格、风格与查询改写。
"""

from __future__ import annotations

from src.llm import get_llm

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
                "将用户问题改写为适合检索医学文献的英文或中英混合查询。"
                "保留核心临床概念（疾病、干预、结局）。只输出改写后的查询，不要解释。"
            ),
        },
        {"role": "user", "content": question},
    ]
    return llm.chat(messages, temperature=0, max_tokens=120).strip() or question


# ---------------------------------------------------------------------------
# 【待完善】临床赛道结构化增强（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def build_clinical_answer_outline(contexts: list[dict]) -> dict:
    """
    【待完善】根据证据生成临床回答大纲。

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
    raise NotImplementedError("待队员实现：build_clinical_answer_outline")


def rank_contexts_for_clinical(contexts: list[dict]) -> list[dict]:
    """
    【待完善】按临床证据优先级对 contexts 重排。

    参数:
        contexts: 原始检索证据。

    返回:
        list[dict]: 重排后的证据列表。

    作用:
        确保指南/荟萃/RCT 更靠前展示与引用。
    """
    raise NotImplementedError("待队员实现：rank_contexts_for_clinical")
