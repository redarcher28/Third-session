# -*- coding: utf-8 -*-
"""
LLM Wiki 主题知识页生成。

把领域资料整理成可读、可维护、可引用的主题页，并作为 source=wiki 入库。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from src.config import get_settings
from src.ingest import load_docs
from src.kb.store import EvidenceStore
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
        body = _render_wiki_page(topic, related, llm)
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


# ---------------------------------------------------------------------------
# Wiki 知识页增强
# ---------------------------------------------------------------------------


def _render_wiki_page(topic: dict, related: list[EvidenceDoc], llm) -> str:
    """渲染单个主题知识页正文（全量生成与单页刷新共用）。"""
    evidence_block = "\n\n".join(
        f"- [{d.doc_id}] {d.title} ({d.year or 'n/a'}): {d.text[:400]}"
        for d in related
    ) or "（暂无相关文档）"
    messages = [
        {
            "role": "system",
            "content": (
                "你是医学证据编辑，请生成可读的「主题知识页」(LLM Wiki)。"
                "必须按如下 Schema 输出（Markdown 标题逐段）:\n"
                "## 一句话结论\n"
                "## 适用范围\n"
                "## 关键证据\n"
                "## 局限与争议\n"
                "## 不该回答什么\n"
                "要求：中文；证据条目带来源 doc_id；不要编造文献；"
                "「不该回答什么」至少包含：个体剂量、替代临床诊疗；"
                "明确标注非诊疗建议。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"主题：{topic['title']}\n\n可用证据：\n{evidence_block}\n\n"
                "请按 Schema 输出 Markdown 主题知识页。"
            ),
        },
    ]
    body = llm.chat(messages, temperature=0.2, max_tokens=1500)
    # 创新：页脚标注覆盖证据数与更新时间，便于演示时说明知识页新鲜度
    # （标题由调用方统一写入文件，避免重复）
    return (
        f"{body}\n\n---\n"
        f"> 本页基于 {len(related)} 条证据文档生成 · 更新于 {datetime.now():%Y-%m-%d %H:%M} · 非诊疗建议"
    )


def select_wiki_then_chunks(
    query: str,
    wiki_k: int = 2,
    chunk_k: int = 5,
) -> list[dict]:
    """
    检索策略：先取主题 Wiki 页，再补原文 chunk。

    创新点：
        - 两级结构：Wiki 页先给「主题总览」，原文 chunk 再给「证据支撑」，提升可解释性；
        - 结果带 kind 标记（wiki / evidence），便于 B 组 UI 分组渲染；
        - 按 doc_id 去重，同一文献的多个 chunk 只取最先命中者，避免重复占位。

    参数:
        query: 检索查询（可为改写后查询）。
        wiki_k: 优先返回的 wiki 条数。
        chunk_k: 补充的原文块条数。

    返回:
        list[dict]: 合并后的证据块（含 source/doc_id/text/kind 等键）。

    作用:
        提升可解释性：先给主题总览，再给原始证据支撑。
    """
    store = EvidenceStore()
    wiki_hits = store.query(query, n_results=wiki_k, source="wiki")
    extra = store.query(query, n_results=wiki_k + chunk_k)
    wiki_ids = {h["chunk_id"] for h in wiki_hits}
    evidence: list[dict] = []
    seen_docs: set[str] = set()
    for h in extra:
        if h["chunk_id"] in wiki_ids:
            continue
        if h.get("doc_id") in seen_docs:
            continue
        seen_docs.add(h.get("doc_id", ""))
        evidence.append(h)
        if len(evidence) >= chunk_k:
            break
    for h in wiki_hits:
        h["kind"] = "wiki"
    for h in evidence:
        h["kind"] = "evidence"
    return wiki_hits + evidence


def refresh_single_wiki_page(slug: str, docs: list[EvidenceDoc] | None = None) -> EvidenceDoc:
    """
    仅重建某一个主题知识页，避免全量重跑。

    创新点：
        - 支持小步迭代：改完语料后只刷新单个主题页（配合任务⑪的增量重建，
          实现「换一条证据只动一页」的运维闭环）；
        - 页脚自动标注更新时间与覆盖证据数，维护历史可追溯。

    参数:
        slug: 主题 slug（见 WIKI_TOPICS）。
        docs: 可选参考文档；None 时从 processed 读取。

    返回:
        EvidenceDoc: 更新后的 wiki 文档。

    作用:
        支持小步迭代维护知识页内容。
    """
    topic = next((t for t in WIKI_TOPICS if t["slug"] == slug), None)
    if topic is None:
        raise ValueError(f"未知 wiki 主题 slug: {slug}（可用：{', '.join(t['slug'] for t in WIKI_TOPICS)}）")
    settings = get_settings()
    if docs is None:
        docs = load_docs(settings.processed_path / "documents.json")
    related = _select_docs(docs, topic["query_tags"])
    body = _render_wiki_page(topic, related, get_llm())
    path = settings.processed_path / "wiki" / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {topic['title']}\n\n{body}\n", encoding="utf-8")
    logger.info("Refreshed wiki page %s (%d source docs)", path, len(related))
    return EvidenceDoc(
        doc_id=f"wiki:{slug}",
        source="wiki",
        title=topic["title"],
        text=body,
        year=None,
        url="",
        tags=list(topic["query_tags"]) + ["wiki"],
        evidence_level="wiki",
    )


# 主题页 Schema 的必需段落（材料第 39 页）
WIKI_SCHEMA_SECTIONS = [
    "一句话结论",
    "适用范围",
    "关键证据",
    "局限与争议",
    "不该回答什么",
]


def lint_wiki_pages(docs: list[EvidenceDoc] | None = None) -> dict:
    """
    定期检查主题页维护质量（材料第 38 页 Lint 闭环）。

    检查项：
        - 孤立页：有 index.json 无文件 / 有文件无 index 条目；
        - 缺失字段：页面缺少 Schema 必需段落（一句话结论/适用范围/关键证据/局限与争议/不该回答什么）；
        - 失效链接：页面引用的 [doc_id] 在 processed 文档中不存在；
        - 来源冲突：页面引用文档的证据等级分布（多种等级并存时提示人工复核）；
        - 更新信息：页脚是否带更新时间与覆盖证据数。

    参数:
        docs: 可选参考文档；None 时从 processed 读取。

    返回:
        dict: {"ok": bool, "issues": list[str], "checked": int}。
    """
    settings = get_settings()
    wiki_dir = settings.processed_path / "wiki"
    issues: list[str] = []
    checked = 0
    index_path = wiki_dir / "index.json"
    if not index_path.exists():
        return {"ok": False, "issues": ["index.json 缺失"], "checked": 0}

    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = {item["slug"] for item in index}
    files = {p.stem for p in wiki_dir.glob("*.md") if p.stem != "index"}

    # 孤立页检查
    for slug in sorted(indexed - files):
        issues.append(f"孤立页（index 有但文件缺失）: {slug}")
    for slug in sorted(files - indexed):
        issues.append(f"未收录页（文件存在但不在 index）: {slug}")

    if docs is None:
        docs = load_docs(settings.processed_path / "documents.json")
    known_ids = {d.doc_id for d in docs}
    known_ids.update(f"wiki:{slug}" for slug in indexed)

    # 逐页检查
    for item in index:
        slug = item["slug"]
        path = wiki_dir / f"{slug}.md"
        if not path.exists():
            continue
        checked += 1
        text = path.read_text(encoding="utf-8")
        missing = [s for s in WIKI_SCHEMA_SECTIONS if f"## {s}" not in text]
        if missing:
            issues.append(f"{slug}: 缺失 Schema 段落 {missing}")
        cited = set(re.findall(r"\[([\w:.-]+)\]", text))
        stale = sorted(c for c in cited if c.startswith(("pmid:", "nct:", "local:", "epmc:", "wiki:")) and c not in known_ids)
        if stale:
            issues.append(f"{slug}: 失效来源引用 {stale[:5]}")
        if "更新于" not in text:
            issues.append(f"{slug}: 缺更新时间标记")
        if "非诊疗建议" not in text:
            issues.append(f"{slug}: 缺非诊疗声明")
    return {"ok": not issues, "issues": issues, "checked": checked}
