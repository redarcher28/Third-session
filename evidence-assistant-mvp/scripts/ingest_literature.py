# -*- coding: utf-8 -*-
"""
把「D1PM_医学知识总结与文献.md」的文献清单解析为结构化 EvidenceDoc。

输出: data/raw/literature.json（doc_id=pmid:XXXX，带 DOI/URL/证据等级），
build_kb 会自动并入采集流程——回答引用 [n] 可锚定真实 PMID。

用法:
    python scripts/ingest_literature.py "D1PM_医学知识总结与文献.md"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

# Allow running as `python scripts/ingest_literature.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings  # noqa: E402
from src.ingest import normalize_evidence_level  # noqa: E402
from src.models import EvidenceDoc  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PMID_RE = re.compile(r"\b(\d{7,8})\b")
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s·|]+)")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_TECH_SKIP = {"RAG", "RAGAS", "MMR", "BM25", "RRF", "arXiv", "LLMWiki", "Karpathy"}


def _clean(cell: str) -> str:
    """清理表格单元格：去 markdown 链接、首尾空白。"""
    cell = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cell)
    return cell.strip()


def parse_literature_tables(md_text: str) -> list[EvidenceDoc]:
    """从 Markdown 表格中解析文献行 → EvidenceDoc 列表。"""
    docs: list[EvidenceDoc] = []
    seen: set[str] = set()
    for line in md_text.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| 证据角色"):
            continue
        cells = [_clean(c) for c in line.strip("|").split("|")]
        joined = " ".join(cells)
        m = _PMID_RE.search(joined)
        if not m or any(k in joined for k in _TECH_SKIP):
            continue
        pmid = m.group(1)
        if pmid in seen:
            continue
        seen.add(pmid)
        doi_m = _DOI_RE.search(joined)
        year_m = _YEAR_RE.search(joined)
        # 标题：取「文献」列（一般含期刊名），截断为可读标题
        title = next((c for c in cells if "et al" in c or "*" in c), joined)
        title = re.sub(r"<[^>]+>", "", title)[:150]
        role_cell = cells[0] if cells else ""
        # 证据等级：结合「证据角色」列（含 RCT/系统综述/队列等字样）+ 标题推断
        level_blob = f"{role_cell} {title} {joined}"
        level = normalize_evidence_level("", level_blob)
        if level == "other":
            lb = level_blob.lower()
            if "meta" in lb or "综述" in lb or "systematic review" in lb:
                level = "meta"
            elif "rct" in lb or "随机对照" in lb:
                level = "rct"
            elif "cohort" in lb or "队列" in lb:
                level = "observational"
            elif "guideline" in lb or "指南" in lb:
                level = "guideline"
        docs.append(
            EvidenceDoc(
                doc_id=f"pmid:{pmid}",
                source="pubmed",
                title=title,
                text=f"文献出处：{title}\n{table_context(line, md_text)}",
                year=int(year_m.group(1)) if year_m else None,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                doi=doi_m.group(1) if doi_m else "",
                tags=["literature", "curated"],
                evidence_level=level,
                journal=next((c for c in cells if "*" in c), ""),
            )
        )
    return docs


def table_context(line: str, md_text: str) -> str:
    """把文献所在表格的上一行（表格标题/证据角色列）作为上下文说明。"""
    lines = md_text.splitlines()
    try:
        idx = lines.index(line)
        for prev in lines[max(0, idx - 6):idx]:
            if "|" in prev and "PMID" not in prev and "文献" in prev and "|---|---" not in prev:
                return prev.strip("|").split("|")[0].strip()
    except ValueError:
        pass
    return "D1PM 课件配套文献（已核实 PMID/DOI）"


def main() -> None:
    parser = argparse.ArgumentParser(description="解析文献清单为 EvidenceDoc")
    parser.add_argument("md", type=Path, help="D1PM_医学知识总结与文献.md 路径")
    args = parser.parse_args()

    text = args.md.read_text(encoding="utf-8")
    docs = parse_literature_tables(text)
    settings = get_settings()
    settings.raw_path.mkdir(parents=True, exist_ok=True)
    out = settings.raw_path / "literature.json"
    out.write_text(
        json.dumps([d.model_dump() for d in docs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("解析出 %d 篇文献 -> %s", len(docs), out)
    for d in docs[:5]:
        print(f"  {d.doc_id} [{d.evidence_level}] {d.title[:50]} ({d.year})")


if __name__ == "__main__":
    main()
