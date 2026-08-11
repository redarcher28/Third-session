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


class Citation(BaseModel):
    """展示给用户的引用/证据卡片结构。"""

    index: int  # 回答中的 [n] 编号
    doc_id: str
    title: str
    source: str
    year: int | None = None
    url: str = ""
    evidence_level: str = "other"
    snippet: str = ""  # 摘要片段，供证据面板展示


class ChatMessage(BaseModel):
    """多轮对话中的一条历史消息。"""

    role: Literal["user", "assistant"] = "user"
    content: str = ""


class AskRequest(BaseModel):
    """HTTP / 内部问答请求体。"""

    question: str  # 用户原问题
    track: Literal["clinical", "nutrition"] = "clinical"  # 赛道
    use_live_tools: bool = False  # 是否启用在线补检索
    top_k: int = 5  # 最终采用的证据条数
    year_from: int | None = None  # 可选：证据起始年份（含）
    year_to: int | None = None  # 可选：证据结束年份（含）
    history: list[ChatMessage] = Field(default_factory=list)  # 可选：之前的对话记录


class AskResponse(BaseModel):
    """问答统一响应（界面与 API 共用）。"""

    answer: str  # 最终回答文本（含伦理声明）
    citations: list[Citation] = Field(default_factory=list)  # 生成引用列表
    contexts: list[Citation] = Field(default_factory=list)  # 证据面板数据
    refused: bool = False  # 是否因证据不足/越界拒答
    rewritten_query: str = ""  # 检索用改写查询
    track: str = "clinical"  # 实际使用的赛道
    citation_check: dict[str, Any] = Field(default_factory=dict)  # 引用校验明细
