# -*- coding: utf-8 -*-
"""
赛道二：健康营养助手 —— 人格、风格与口语归一改写。
"""

from __future__ import annotations

from src.llm import get_llm

# 面向普通消费者的系统人格
NUTRITION_PERSONA = (
    "你是面向普通消费者的「健康营养助手」。"
    "把专业证据转成可理解的科普，语气友好、可执行，强调非诊疗。"
    "避免开药、给剂量或替代医生就诊。"
)

# 科普回答结构约束
NUTRITION_STYLE = (
    "结构：通俗结论 → 证据一句话 → 你可以怎么做 → 何时就医；"
    "少用术语，必要时括号解释；引用仍用 [n]。"
)

# 营养赛道检索时加权的标签
BOOST_TAGS = ["diet", "mediterranean", "hypertension", "hyperlipidemia", "diabetes"]


def rewrite_nutrition_query(question: str) -> str:
    """
    将消费者口语问题改写为可检索的证据查询。

    参数:
        question: 用户原问题。

    返回:
        str: 改写后查询；失败回退原问题。
    """
    llm = get_llm()
    messages = [
        {
            "role": "system",
            "content": (
                "把消费者口语健康问题改写成可检索的证据查询"
                "（可含饮食模式、营养干预、疾病风险关键词）。"
                "只输出改写查询，不要解释。"
            ),
        },
        {"role": "user", "content": question},
    ]
    return llm.chat(messages, temperature=0, max_tokens=120).strip() or question


# ---------------------------------------------------------------------------
# 【待完善】营养赛道科普增强（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def build_nutrition_action_tips(contexts: list[dict]) -> list[str]:
    """
    【待完善】从证据中提炼面向消费者的可执行建议条目。

    参数:
        contexts: 检索证据列表。

    返回:
        list[str]: 3~5 条「你可以怎么做」提示（禁止开药剂量）。

    作用:
        强化营养赛道可读性与行动性，同时保持可追溯。
    """
    raise NotImplementedError("待队员实现：build_nutrition_action_tips")


def simplify_medical_terms(text: str) -> str:
    """
    【待完善】把回答中的专业术语改写成更通俗的说法（可附括号注释）。

    参数:
        text: 待改写文本。

    返回:
        str: 通俗化后的文本。

    作用:
        拉大与临床赛道的可读性差异，突出赛道二产品定位。
    """
    raise NotImplementedError("待队员实现：simplify_medical_terms")
