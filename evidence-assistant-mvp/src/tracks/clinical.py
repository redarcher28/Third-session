# -*- coding: utf-8 -*-
"""
赛道一：临床证据助手 —— 人格、风格与查询改写。
"""

from __future__ import annotations

from src.llm import get_llm
from src.tracks.prompt_profiles import build_query_messages, get_track_profile

_PROFILE = get_track_profile("clinical")

# 保留这些公开常量，兼容已有 pipeline / 评测脚本；实际内容统一由 Prompt 配置管理。
CLINICAL_PERSONA = _PROFILE.persona
CLINICAL_STYLE = _PROFILE.style
PREFER_LEVELS = list(_PROFILE.prefer_levels)


def rewrite_clinical_query(question: str) -> str:
    """
    将用户临床问题改写为更适合文献检索的查询。

    参数:
        question: 用户原问题（中文或中英混合）。

    返回:
        str: 改写后的检索查询；失败时回退为原问题。
    """
    llm = get_llm()
    messages = build_query_messages("clinical", question)
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
