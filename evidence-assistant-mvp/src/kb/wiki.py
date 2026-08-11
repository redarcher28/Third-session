# -*- coding: utf-8 -*-
"""
LLM Wiki 主题知识页生成。

把领域资料整理成可读、可维护、可引用的主题页，并作为 source=wiki 入库。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import get_settings
from src.ingest import load_docs
from src.llm import get_llm
from src.models import EvidenceDoc

logger = logging.getLogger(__name__)

# 预设主题：标题 + 用于挑选相关文档的标签
WIKI_TOPICS = [
    {
        "slug": "hypertension-long-term-meds",
        "title": "高血压患者为什么需要长期服药",
        "query_tags": ["hypertension", "guideline"],
    },
    {
        "slug": "lipid-lifestyle-vs-drugs",
        "title": "血脂偏高：生活方式干预与药物治疗的证据",
        "query_tags": ["hyperlipidemia", "diet"],
    },
    {
        "slug": "mediterranean-diet-cvd",
        "title": "地中海饮食与心血管风险",
        "query_tags": ["mediterranean", "diet", "cardiovascular"],
    },
    {
        "slug": "sodium-hypertension",
        "title": "限钠饮食对高血压的帮助",
        "query_tags": ["hypertension", "diet"],
    },
    {
        "slug": "diabetes-diet",
        "title": "糖尿病饮食干预与心血管风险",
        "query_tags": ["diabetes", "diet"],
    },
    {
        "slug": "dash-diet",
        "title": "DASH饮食与血压管理",
        "query_tags": ["hypertension", "diet"],
    },
    {
        "slug": "statin-evidence",
        "title": "他汀类药物降低心血管事件的证据要点",
        "query_tags": ["hyperlipidemia", "cardiovascular"],
    },
    {
        "slug": "lifestyle-first",
        "title": "三高管理中的生活方式干预总览",
        "query_tags": ["diet", "hypertension", "hyperlipidemia", "diabetes"],
    },
]


def _select_docs(docs: list[EvidenceDoc], tags: list[str], limit: int = 8) -> list[EvidenceDoc]:
    """按标签重合度挑选与主题最相关的文档。"""
    scored: list[tuple[int, EvidenceDoc]] = []
    for d in docs:
        score = len(set(d.tags) & set(tags))
        if score:
            scored.append((score, d))
    scored.sort(key=lambda x: (-x[0], -(x[1].year or 0)))
    return [d for _, d in scored[:limit]]


def generate_wiki_pages(docs: list[EvidenceDoc] | None = None) -> list[EvidenceDoc]:
    """
    为预设主题生成 Markdown 知识页，并返回可入库的 Wiki 文档。

    参数:
        docs: 参考文档；None 时从 processed/documents.json 读取。

    返回:
        list[EvidenceDoc]: source=wiki 的主题页文档列表。
    """
    settings = get_settings()
    wiki_dir = settings.processed_path / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    if docs is None:
        docs = load_docs(settings.processed_path / "documents.json")
    llm = get_llm()
    wiki_docs: list[EvidenceDoc] = []

    for topic in WIKI_TOPICS:
        related = _select_docs(docs, topic["query_tags"])
        evidence_block = "\n\n".join(
            f"- [{d.doc_id}] {d.title} ({d.year or 'n/a'}): {d.text[:400]}"
            for d in related
        ) or "（暂无相关文档）"
        messages = [
            {
                "role": "system",
                "content": (
                    "你是医学证据编辑，请生成可读的「主题知识页」(LLM Wiki)。"
                    "要求：中文；包含主张要点、证据条目、引用列表；不要编造文献；"
                    "明确标注非诊疗建议。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"主题：{topic['title']}\n\n可用证据：\n{evidence_block}\n\n"
                    "请输出 Markdown 主题知识页。"
                ),
            },
        ]
        body = llm.chat(messages, temperature=0.2, max_tokens=1500)
        path = wiki_dir / f"{topic['slug']}.md"
        path.write_text(f"# {topic['title']}\n\n{body}\n", encoding="utf-8")
        wiki_docs.append(
            EvidenceDoc(
                doc_id=f"wiki:{topic['slug']}",
                source="wiki",
                title=topic["title"],
                text=body,
                year=None,
                url="",
                tags=list(topic["query_tags"]) + ["wiki"],
                evidence_level="wiki",
            )
        )
        logger.info("Wrote wiki page %s", path)

    index = [{"slug": t["slug"], "title": t["title"]} for t in WIKI_TOPICS]
    (wiki_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return wiki_docs


def _build_single_wiki_doc(topic: dict, docs: list[EvidenceDoc], wiki_dir: Path) -> EvidenceDoc:
    """按一个主题生成 Wiki 文档，并写入对应 Markdown。"""
    llm = get_llm()
    related = _select_docs(docs, topic["query_tags"])
    evidence_block = "\n\n".join(
        f"- [{d.doc_id}] {d.title} ({d.year or 'n/a'}): {d.text[:400]}"
        for d in related
    ) or "（暂无相关文档）"
    messages = [
        {
            "role": "system",
            "content": (
                "你是医学证据编辑，请生成可读的「主题知识页」(LLM Wiki)。"
                "要求：中文；包含主张要点、证据条目、引用列表；不要编造文献；"
                "明确标注非诊疗建议。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"主题：{topic['title']}\n\n可用证据：\n{evidence_block}\n\n"
                "请输出 Markdown 主题知识页。"
            ),
        },
    ]
    body = llm.chat(messages, temperature=0.2, max_tokens=1500)
    path = wiki_dir / f"{topic['slug']}.md"
    path.write_text(f"# {topic['title']}\n\n{body}\n", encoding="utf-8")
    logger.info("Wrote wiki page %s", path)
    return EvidenceDoc(
        doc_id=f"wiki:{topic['slug']}",
        source="wiki",
        title=topic["title"],
        text=body,
        year=None,
        url="",
        tags=list(topic["query_tags"]) + ["wiki"],
        evidence_level="wiki",
    )


# ---------------------------------------------------------------------------
# 【待完善】Wiki 知识页增强（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def select_wiki_then_chunks(
    query: str,
    wiki_k: int = 2,
    chunk_k: int = 5,
) -> list[dict]:
    """
    【待完善】检索策略：先取主题 Wiki 页，再补原文 chunk。

    参数:
        query: 检索查询（可为改写后查询）。
        wiki_k: 优先返回的 wiki 条数。
        chunk_k: 补充的原文块条数。

    返回:
        list[dict]: 合并后的证据块（含 source/doc_id/text 等键）。

    作用:
        提升可解释性：先给主题总览，再给原始证据支撑。
    """
    from src.kb.store import EvidenceStore

    store = EvidenceStore()
    n_results = max(wiki_k + chunk_k, 8)
    hits = store.query(query, n_results=n_results)
    wiki_hits = [h for h in hits if str(h.get("source")) == "wiki"][:wiki_k]
    chunk_hits = [h for h in hits if str(h.get("source")) != "wiki"][:chunk_k]

    out: list[dict] = []
    seen: set[str] = set()
    for item in wiki_hits + chunk_hits:
        key = str(item.get("doc_id") or item.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def refresh_single_wiki_page(slug: str, docs: list[EvidenceDoc] | None = None) -> EvidenceDoc:
    """
    【待完善】仅重建某一个主题知识页，避免全量重跑。

    参数:
        slug: 主题 slug（见 WIKI_TOPICS）。
        docs: 可选参考文档；None 时从 processed 读取。

    返回:
        EvidenceDoc: 更新后的 wiki 文档。

    作用:
        支持小步迭代维护知识页内容。
    """
    settings = get_settings()
    wiki_dir = settings.processed_path / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    if docs is None:
        docs = load_docs(settings.processed_path / "documents.json")

    topic = next((t for t in WIKI_TOPICS if t["slug"] == slug), None)
    if topic is None:
        known = ", ".join(t["slug"] for t in WIKI_TOPICS)
        raise ValueError(f"Unknown wiki slug: {slug}. Known slugs: {known}")

    doc = _build_single_wiki_doc(topic, docs, wiki_dir)
    index = [{"slug": t["slug"], "title": t["title"]} for t in WIKI_TOPICS]
    (wiki_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return doc
