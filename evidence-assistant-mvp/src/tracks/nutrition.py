# -*- coding: utf-8 -*-
"""
赛道二：健康营养助手 —— 人格、风格与口语归一改写。

产品边界：面向普通消费者，只谈饮食与生活方式的一般建议；
绝不输出具体药量/剂量/服药方案，此类问题直接通俗拒答。
"""

from __future__ import annotations

import re

from src.llm import get_llm
from src.tracks.prompt_profiles import build_query_messages, get_track_profile

_PROFILE = get_track_profile("nutrition")

# 兼容已有 pipeline / 评测脚本；实际内容统一由 Prompt 配置管理。
NUTRITION_PERSONA = _PROFILE.persona
NUTRITION_STYLE = _PROFILE.style
BOOST_TAGS = list(_PROFILE.boost_tags)

# 中文消费者问法 -> 英文医学检索锚点。用于离线/弱 embedding 场景下连接中文问题与英文文献摘要。
_NUTRITION_QUERY_ALIASES = [
    (("膳食纤维", "纤维", "全谷", "全麦", "粗粮"), "dietary fiber whole grains cardiovascular risk"),
    (("含糖饮料", "甜饮料", "奶茶", "可乐", "饮料含糖"), "sugar-sweetened beverages diabetes cardiovascular risk"),
    (("超加工", "加工食品", "加工零食"), "ultra-processed foods cardiovascular diabetes"),
    (("植物性", "植物基", "素食"), "plant-based diet vegetarian diet cardiovascular"),
    (("低碳", "低碳水", "生酮"), "low-carbohydrate diet type 2 diabetes glycemic control"),
    (("间歇性禁食", "轻断食", "断食"), "intermittent fasting diabetes weight loss"),
    (("坚果",), "nuts cardiovascular risk"),
    (("豆类", "豆制品", "豆子"), "legumes cardiovascular diabetes"),
    (("鱼油", "欧米伽", "omega", "omega-3"), "omega-3 fatty acids cardiovascular"),
    (("地中海",), "Mediterranean diet cardiovascular risk"),
    (("dash", "得舒"), "DASH diet blood pressure hypertension"),
    (("限钠", "低钠", "少盐", "盐"), "dietary sodium blood pressure hypertension"),
    (("肥胖", "减重", "减肥", "体重"), "obesity weight loss dietary intervention cardiovascular risk"),
]

# 问药量/剂量时的通俗拒答（面向普通群众的口吻）
NUTRITION_DOSAGE_REFUSAL = (
    "这个问题涉及具体用药剂量，我不能替你开处方或给用药方案。\n\n"
    "如果你在问自己或家人的服药问题：请带上检查报告和正在吃的药，"
    "咨询执业医师或药师，不要自行增减药量。\n"
    "我可以帮你了解饮食与生活方式管理的一般知识，"
    "比如血压高的人饮食上可以注意什么、地中海饮食怎么搭配。"
)

# 问药量/剂量/停药/减药的问题关键词：命中即通俗拒答（确定性规则，不依赖 LLM）
# 对应材料规则 4「问吃药/停药 → 安全拒答」；克/g 不在此列——
# 「每天吃几克盐」属饮食建议可正常回答，药量禁区以 mg/片/粒/单位/停药减药为主
_DOSAGE_PATTERNS = [
    r"药量|剂量|服药方案|怎么吃(这|那个)?药|用药量",
    r"吃(多少|几)药",
    r"(几片|几粒|多少片|多少粒|每天\d+片|一次\d+粒)",
    r"\d+\s*(mg|毫克|片|粒|单位)\s*(怎么|多少|几次|服用|吃)",
    r"(胰岛素|降糖药|降压药|他汀).{0,12}(多少|几|剂量|用量|单位)",
    r"一天(吃|服|用)(几|多少|几次|\d+次|\d+片|\d+粒)",
    r"(停|减|换|加)药|药量(减半|调整|减少)|停药|减量|增量|加量",
    r"能(不能)?(停|减|换)药",
    r"药.{0,2}(吃|服)(多久|几天|多长时间|能不能停)",
]

# 医学术语 → 通俗说法（长词在前，避免子串误替换）
_TERM_MAP = {
    "低密度脂蛋白胆固醇（LDL-C）": "低密度脂蛋白胆固醇（俗称「坏胆固醇」）",
    "高密度脂蛋白胆固醇（HDL-C）": "高密度脂蛋白胆固醇（俗称「好胆固醇」）",
    "低密度脂蛋白胆固醇": "低密度脂蛋白胆固醇（「坏胆固醇」）",
    "高密度脂蛋白胆固醇": "高密度脂蛋白胆固醇（「好胆固醇」）",
    "LDL-C": "「坏胆固醇」",
    "HDL-C": "「好胆固醇」",
    "收缩压": "收缩压（血压读数里的「高压」）",
    "舒张压": "舒张压（血压读数里的「低压」）",
    "随机对照试验": "随机对照试验（严格对比效果的临床试验，证据强度较高）",
    "荟萃分析": "荟萃分析（把多项研究结果合并统计，结论更稳）",
    "系统性综述": "系统性综述（系统收集某主题全部研究的综述）",
    "systematic review": "系统性综述（系统收集某主题全部研究的综述）",
    "meta-analysis": "荟萃分析（把多项研究结果合并统计）",
    "甘油三酯": "甘油三酯（血液里的一种脂肪）",
    "心脑血管事件": "心脑血管事件（如心梗、中风等严重问题）",
    "心血管事件": "心血管事件（如心梗、中风等严重问题）",
    "心血管结局": "心血管结局（如心梗、中风等终点事件）",
    "心血管风险": "心血管风险（发生心梗、中风等问题的风险）",
    "饱和脂肪": "饱和脂肪（肥肉、动物油里较多的脂肪）",
    "反式脂肪": "反式脂肪（油炸食品、加工零食里常见的坏脂肪）",
    "依从性": "依从性（能否长期坚持）",
    "膳食纤维": "膳食纤维（蔬果全谷物里的粗纤维）",
    "危险因素": "危险因素（增加患病风险的因素）",
}

# 「你可以怎么做」建议句的动作词与禁止词
_ACTION_WORDS = ["建议", "推荐", "应", "可", "避免", "减少", "增加", "限制", "坚持", "选择", "注意", "有助于", "需要"]
_BANNED_IN_TIP = ["剂量", "毫克", "mg", "服药", "吃药", "用药", "几片", "几粒", "单位"]


def rewrite_nutrition_query(question: str) -> str:
    """
    将消费者口语问题改写为可检索的证据查询。

    参数:
        question: 用户原问题。

    返回:
        str: 改写后查询；失败回退原问题。
    """
    llm = get_llm()
    messages = build_query_messages("nutrition", question)
    rewritten = llm.chat(messages, temperature=0, max_tokens=120).strip() or question
    return append_nutrition_query_aliases(question, rewritten)


def append_nutrition_query_aliases(question: str, rewritten: str | None = None) -> str:
    """
    为中文营养问题追加英文检索锚点，增强中文问题对英文医学证据的召回。

    参数:
        question: 用户原问题。
        rewritten: LLM 改写后的查询；None 时使用原问题。

    返回:
        str: 原查询 + 命中的英文医学关键词。

    作用:
        赛道二用户多为中文口语提问，而 PubMed/Europe PMC 摘要多为英文；
        该映射只添加主题检索词，不生成任何文献链接或证据结论。
    """
    base = (rewritten or question).strip()
    lower = question.lower()
    aliases: list[str] = []
    seen: set[str] = set()
    for keys, phrase in _NUTRITION_QUERY_ALIASES:
        if any(k.lower() in lower for k in keys) and phrase not in seen:
            aliases.append(phrase)
            seen.add(phrase)
    if not aliases:
        return base
    return f"{base} {' '.join(aliases)}"[:500]


def detect_dosage_request(question: str) -> bool:
    """
    判断问题是否在索要具体药量/剂量（确定性规则，不依赖 LLM）。

    参数:
        question: 用户原问题。

    返回:
        bool: True 表示应通俗拒答并转介医生/药师。

    作用:
        产品边界硬拦截：赛道二绝不输出药量，防止演示事故。
    """
    q = question.lower()
    return any(re.search(p, q) for p in _DOSAGE_PATTERNS)


def simplify_medical_terms(text: str) -> str:
    """
    把回答中的专业术语改写成更通俗的说法（可附括号注释）。

    参数:
        text: 待改写文本。

    返回:
        str: 通俗化后的文本。

    作用:
        拉大与临床赛道的可读性差异，突出赛道二产品定位。
    """
    out = text
    for term, plain in _TERM_MAP.items():
        if term in out:
            out = out.replace(term, plain)
    return out


def flag_dosage_in_answer(text: str) -> bool:
    """
    防御性检查：生成结果是否仍含具体药量表述。

    参数:
        text: 生成后的回答文本。

    返回:
        bool: True 表示需要追加「以医生/药师意见为准」警示。
    """
    if not re.search(r"\d+(\.\d+)?\s*(mg|毫克|克|g|片|粒|单位)", text):
        return False
    return any(k in text for k in ("服药", "吃药", "用药", "剂量", "每次", "每天服用", "每日服用"))


def build_nutrition_action_tips(contexts: list[dict]) -> list[str]:
    """
    从证据中提炼面向消费者的可执行建议条目。

    创新点：
        - 动作句提取：只保留含「建议/应/避免/减少/限制/坚持」等动作词的句子；
        - 剂量禁入：含药量表述的句子一律剔除，守住产品边界；
        - 去重 + 截断，稳定输出 3~5 条「你可以怎么做」。

    参数:
        contexts: 检索证据列表。

    返回:
        list[str]: 3~5 条「你可以怎么做」提示（禁止开药剂量）。

    作用:
        强化营养赛道可读性与行动性，同时保持可追溯。
    """
    tips: list[str] = []
    seen: set[str] = set()
    for c in contexts:
        text = str(c.get("text") or "")
        for sent in re.split(r"(?<=[。！？!?；;])", text):
            sent = sent.strip()
            if not sent or any(w in sent for w in _BANNED_IN_TIP):
                continue
            if not any(w in sent for w in _ACTION_WORDS):
                continue
            key = re.sub(r"\s+", "", sent)[:60]
            if key in seen:
                continue
            seen.add(key)
            tips.append(sent[:120])
            if len(tips) >= 5:
                return tips
    return tips
