# -*- coding: utf-8 -*-
"""
本地文档采集：种子摘要、PDF、Markdown。

无外网时仅靠种子语料即可跑通演示。
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import get_settings
from src.models import EvidenceDoc

logger = logging.getLogger(__name__)


# 窄领域种子语料（心脑血管 / 血脂 / 高血压 / 饮食），仅作演示，非诊疗依据
SEED_DOCS: list[dict] = [
    {
        "doc_id": "local:seed-htn-guideline",
        "source": "local",
        "title": "高血压长期药物治疗的循证要点（样例摘要）",
        "text": (
            "高血压是慢性病，多数患者需要长期甚至终身药物治疗以持续控制血压并降低"
            "心脑血管事件风险。指南普遍建议：确诊高血压后，在生活方式干预基础上，"
            "根据血压水平与合并症选择降压药；达到目标血压后通常需维持治疗，"
            "擅自停药可能导致血压反弹与事件风险上升。生活方式干预包括限盐、"
            "减重、规律运动、限制饮酒等，可增强药效或减少用药剂量，但多数中高危"
            "患者仍需药物维持。证据来源包括大型结局试验与国内外高血压管理指南摘要。"
        ),
        "year": 2023,
        "url": "",
        "tags": ["hypertension", "guideline", "cardiovascular"],
        "evidence_level": "guideline",
    },
    {
        "doc_id": "local:seed-lipid-lifestyle-drug",
        "source": "local",
        "title": "血脂异常：生活方式与药物治疗证据对比（样例）",
        "text": (
            "体检发现血脂偏高时，指南通常首先评估总体心血管风险。生活方式干预"
            "（减少饱和脂肪与反式脂肪、增加膳食纤维、减重、运动）可改善LDL-C与"
            "甘油三酯，对低危人群可作为一线措施。中高危或LDL-C显著升高者，"
            "他汀类药物有明确的心血管结局获益证据，可显著降低主要不良心血管事件。"
            "生活方式与药物并非互斥：即使启动药物，仍应坚持饮食与运动干预。"
            "是否用药需结合风险分层与医患共同决策，本摘要不作个体处方。"
        ),
        "year": 2022,
        "url": "",
        "tags": ["hyperlipidemia", "diet", "guideline", "cardiovascular"],
        "evidence_level": "guideline",
    },
    {
        "doc_id": "local:seed-mediterranean",
        "source": "local",
        "title": "地中海饮食与心血管风险：证据摘要（样例）",
        "text": (
            "地中海饮食强调橄榄油、坚果、蔬菜水果、全谷物、鱼类，限制红肉与加工食品。"
            "多项随机对照试验与荟萃分析显示，坚持地中海饮食模式与较低的心血管事件"
            "风险相关，尤其在已有高危因素的人群中。机制可能包括改善血脂谱、血压、"
            "炎症与内皮功能。证据强度整体优于单一营养素补充。个体效果受依从性与"
            "基线饮食影响，不能替代必要的药物治疗。"
        ),
        "year": 2021,
        "url": "",
        "tags": ["mediterranean", "diet", "cardiovascular", "meta"],
        "evidence_level": "meta",
    },
    {
        "doc_id": "local:seed-sodium",
        "source": "local",
        "title": "限钠饮食对高血压的作用（样例证据摘要）",
        "text": (
            "系统评价与临床试验表明，减少膳食钠摄入可降低收缩压与舒张压，"
            "对高血压患者及血压正常偏高者均有益，效应呈剂量相关。"
            "常见建议为每日钠摄入控制在一定范围内（具体数值以当地指南为准），"
            "同时注意整体饮食质量。限钠常与DASH或地中海饮食模式结合效果更好。"
            "极度限盐需关注特殊人群（如某些肾病）的个体化评估。"
        ),
        "year": 2020,
        "url": "",
        "tags": ["hypertension", "diet", "guideline"],
        "evidence_level": "meta",
    },
    {
        "doc_id": "local:seed-diabetes-diet",
        "source": "local",
        "title": "2型糖尿病饮食干预与心血管风险（样例）",
        "text": (
            "饮食模式干预是2型糖尿病管理的基础。减少精制碳水、控制总能量、"
            "选择优质脂肪与高纤维食物有助于改善血糖与体重。部分研究显示结构化"
            "营养干预可降低心血管危险因素。药物（包括降糖与降脂/降压）在中高危"
            "患者中仍关键。营养建议应个体化，并强调可持续性与低血糖风险防范。"
        ),
        "year": 2022,
        "url": "",
        "tags": ["diabetes", "diet", "cardiovascular"],
        "evidence_level": "guideline",
    },
    {
        "doc_id": "local:seed-dash",
        "source": "local",
        "title": "DASH饮食模式与血压（样例）",
        "text": (
            "DASH（Dietary Approaches to Stop Hypertension）饮食富含蔬果、低脂乳制品、"
            "全谷物，限制饱和脂肪与甜食。临床试验证实DASH可显著降低血压，"
            "与限钠联用时降压幅度更大。该模式也被用于血脂与代谢健康的综合管理。"
        ),
        "year": 2019,
        "url": "",
        "tags": ["hypertension", "diet", "rct"],
        "evidence_level": "rct",
    },
]


def load_pdf_as_docs(pdf_path: Path) -> list[EvidenceDoc]:
    """
    将单个 PDF 转为 EvidenceDoc（依赖 pymupdf4llm）。

    参数:
        pdf_path: PDF 文件路径。

    返回:
        list[EvidenceDoc]: 通常 0 或 1 条；解析失败返回空列表。
    """
    try:
        import pymupdf4llm
    except ImportError:
        logger.warning("pymupdf4llm not installed; skip PDF %s", pdf_path)
        return []
    md = pymupdf4llm.to_markdown(str(pdf_path))
    title = pdf_path.stem
    return [
        EvidenceDoc(
            doc_id=f"local:pdf:{pdf_path.stem}",
            source="local",
            title=title,
            text=md[:20000],
            year=None,
            url="",
            tags=["ebook", "local"],
            evidence_level="ebook",
        )
    ]


def ingest_local(include_seed: bool = True) -> list[EvidenceDoc]:
    """
    采集本地种子 + data/raw/local 下的 PDF/Markdown。

    参数:
        include_seed: 是否包含内置 SEED_DOCS。

    返回:
        list[EvidenceDoc]: 本地证据文档列表。
    """
    settings = get_settings()
    local_dir = settings.raw_path / "local"
    local_dir.mkdir(parents=True, exist_ok=True)
    docs: list[EvidenceDoc] = []
    if include_seed:
        docs.extend(EvidenceDoc.model_validate(d) for d in SEED_DOCS)
    for pdf in local_dir.glob("*.pdf"):
        docs.extend(load_pdf_as_docs(pdf))
    for md_file in local_dir.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        docs.append(
            EvidenceDoc(
                doc_id=f"local:md:{md_file.stem}",
                source="local",
                title=md_file.stem,
                text=text[:20000],
                tags=["local"],
                evidence_level="ebook",
            )
        )
    logger.info("Local ingest -> %d docs", len(docs))
    return docs


# ---------------------------------------------------------------------------
# 【待完善】本地语料增强（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def enrich_local_doc_tags(doc: EvidenceDoc) -> EvidenceDoc:
    """
    【待完善】根据标题与正文自动补全本地文档的 tags / evidence_level。

    参数:
        doc: 本地 EvidenceDoc（种子、PDF 或 Markdown）。

    返回:
        EvidenceDoc: 补全标签后的新文档（或原地更新后的同一对象）。

    作用:
        提升本地语料在混合检索中的可过滤性与加权效果。
    """
    raise NotImplementedError("待队员实现：enrich_local_doc_tags")


def split_long_local_markdown(md_path: Path, max_chars: int = 8000) -> list[EvidenceDoc]:
    """
    【待完善】将超长本地 Markdown 按标题层级切成多条 EvidenceDoc。

    参数:
        md_path: Markdown 文件路径。
        max_chars: 单条文档正文上限。

    返回:
        list[EvidenceDoc]: 切分后的本地文档列表。

    作用:
        避免整本电子书塞进单条记录，改善切块与检索粒度。
    """
    raise NotImplementedError("待队员实现：split_long_local_markdown")
