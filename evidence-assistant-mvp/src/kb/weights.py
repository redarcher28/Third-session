# -*- coding: utf-8 -*-
"""
证据优先级权重：为 B 组检索加权/重排提供统一口径。

A 组产出：B 组的 score_evidence_priority / filter_by_year_range 等
待完善任务可直接调用本模块，避免各自定义权重导致口径不一致。
"""

from __future__ import annotations

from datetime import datetime

# 证据等级 → 基础权重（与 src/models.py 的 EvidenceLevel 枚举一一对应）
EVIDENCE_LEVEL_WEIGHTS: dict[str, float] = {
    "meta": 1.0,  # 荟萃/系统综述：证据强度最高
    "guideline": 0.95,  # 指南：综合多源
    "rct": 0.9,  # 随机对照试验
    "wiki": 0.8,  # 主题知识页（总览优先）
    "observational": 0.6,  # 观察性研究
    "ebook": 0.4,  # 电子书/本地资料
    "other": 0.3,  # 未知
}


def evidence_priority(level: str) -> float:
    """按证据等级取基础权重，未知等级给最低分。"""
    return EVIDENCE_LEVEL_WEIGHTS.get(level, EVIDENCE_LEVEL_WEIGHTS["other"])


def recency_weight(
    year: int | None,
    half_life: int = 8,
    ref_year: int | None = None,
) -> float:
    """
    时效衰减：越新的文献权重越高，旧文献按半衰期衰减。

    参数:
        year: 发表年份；None 给中性 0.5（不偏袒也不惩罚）。
        half_life: 权重减半所需的年数，默认 8 年。
        ref_year: 基准年，默认当前年份（测试时可固定）。

    返回:
        float: 0~1 之间的时效权重。
    """
    if year is None:
        return 0.5
    ref = ref_year or datetime.now().year
    return 0.5 ** (max(0, ref - year) / half_life)


def combined_priority(
    level: str,
    year: int | None,
    level_w: float = 1.0,
    recency_w: float = 0.5,
) -> float:
    """
    组合优先级 = 等级分 × level_w + 时效分 × recency_w。

    供 B 组 score_evidence_priority 直接调用：
        分越高越优先，临床赛道突出「新 + 高等级证据」。

    参数:
        level: 证据等级（EvidenceLevel 枚举值）。
        year: 发表年份。
        level_w: 等级权重系数。
        recency_w: 时效权重系数。
    """
    return evidence_priority(level) * level_w + recency_weight(year) * recency_w
