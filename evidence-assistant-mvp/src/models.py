# -*- coding: utf-8 -*-
"""
统一数据模型（全链路接口对齐用）。

各模块之间传递证据、切块、引用、问答请求/响应时，优先使用本文件中的模型。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# 证据等级：随机对照 / 荟萃 / 指南 / 观察性 / 电子书 / Wiki / 其他
EvidenceLevel = Literal[
    "rct", "meta", "guideline", "observational", "ebook", "wiki", "other"
]
# 数据来源名称
SourceName = Literal["pubmed", "clinicaltrials", "europepmc", "local", "wiki"]
# 记录类型：区分发表文献、试验注册、指南摘录等，避免「注册=RCT」混淆
RecordType = Literal[
    "published_article",
    "trial_registry",
    "guideline_excerpt",
    "local_doc",
    "wiki_page",
    "other",
]


class EvidenceDoc(BaseModel):
    """采集层统一文档结构（入库前的完整文献/试验/本地摘要）。"""

    doc_id: str  # 全局唯一，如 pmid:123 / nct:NCT... / wiki:slug
    source: SourceName  # 来源系统
    title: str  # 标题
    text: str  # 正文或摘要
    year: int | None = None  # 发表年，未知则为 None
    url: str = ""  # 可点击溯源链接
    tags: list[str] = Field(default_factory=list)  # 主题标签，如 hypertension
    evidence_level: EvidenceLevel = "other"  # 证据等级
    journal: str = ""  # 期刊名（可选）
    doi: str = ""  # DOI（可选）
    record_type: RecordType = "other"  # 记录语义类型
    citation_eligible: bool = True  # 是否可作为回答中的循证引用
    source_locator: str = ""  # 稳定溯源键（URL 或 doc_id）
    extra: dict[str, Any] = Field(default_factory=dict)  # 扩展字段


class Chunk(BaseModel):
    """切分后的检索单元，必须保留可追溯元数据。"""

    chunk_id: str  # 块 ID，建议 doc_id#c0
    doc_id: str  # 所属文档 ID
    source: SourceName
    title: str
    text: str  # 块文本
    year: int | None = None
    url: str = ""
    tags: list[str] = Field(default_factory=list)
    evidence_level: EvidenceLevel = "other"
    chunk_index: int = 0  # 在原文档中的块序号
    record_type: RecordType = "other"
    citation_eligible: bool = True
    source_locator: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)  # 来自 EvidenceDoc.extra 的检索侧车


class Citation(BaseModel):
    """展示给用户的引用/证据卡片结构。"""

    index: int  # 回答中的 [n] 编号
    doc_id: str
    title: str
    source: str
    year: int | None = None
    url: str = ""
    evidence_level: str = "other"
    text: str = ""  # 检索块完整正文，供证据面板展示
    snippet: str = ""  # 句子对齐的短摘要，供参考文献等紧凑场景
    record_type: str = "other"  # 记录语义类型（试验注册 vs 发表文献等）
    trial_status: str = ""  # 临床试验状态（仅 trial_registry）


class AskRequest(BaseModel):
    """HTTP / 内部问答请求体。"""

    question: str = Field(min_length=1)  # 用户原问题
    track: Literal["clinical", "nutrition"] = "clinical"  # 赛道
    use_live_tools: bool = False  # 是否启用在线补检索
    top_k: int = Field(default=5, ge=3, le=8)  # 最终采用的证据条数


class AskResponse(BaseModel):
    """问答统一响应（界面与 API 共用）。"""

    answer: str  # 最终回答文本（含伦理声明）
    citations: list[Citation] = Field(default_factory=list)  # 生成引用列表
    contexts: list[Citation] = Field(default_factory=list)  # 证据面板数据
    refused: bool = False  # 是否因证据不足/越界拒答
    rewritten_query: str = ""  # 检索用改写查询
    track: str = "clinical"  # 实际使用的赛道
    prompt_version: str = ""  # 赛道 Prompt 栈版本
    retrieval: dict[str, Any] = Field(default_factory=dict)  # 检索摘要与来源分布
    citation_check: dict[str, Any] = Field(default_factory=dict)  # 引用校验明细
    timings_ms: dict[str, float] = Field(default_factory=dict)  # 各阶段耗时，便于定位慢点
