# -*- coding: utf-8 -*-
"""
批量入库 500-collection：CSV 清单 + PDF 全文 → 结构化 EvidenceDoc。

输出: data/raw/collection_500.json（doc_id=pmid:XXXX，带影响因子等元数据），
build_kb 会自动并入采集流程。

用法:
    python scripts/ingest_pdf_collection.py <目录> <manifest.csv>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/ingest_pdf_collection.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.ingest import normalize_evidence_level  # noqa: E402
from src.models import EvidenceDoc  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 8000  # 与现有采集口径一致：正文截断上限


def extract_pdf_text(pdf_path: Path) -> str:
    """提取 PDF 全文（有文字层），截断到 MAX_TEXT_CHARS。"""
    doc = pymupdf.open(str(pdf_path))
    parts = [page.get_text() for page in doc]
    return "\n".join(parts)[:MAX_TEXT_CHARS]


def ingest_collection(data_dir: Path, manifest: Path) -> int:
    """按 manifest 批量读取 PDF 并转 EvidenceDoc，保存为 collection_500.json。"""
    settings = get_settings()
    settings.raw_path.mkdir(parents=True, exist_ok=True)
    docs: list[EvidenceDoc] = []
    skipped = 0
    with manifest.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = (row.get("pmid") or "").strip()
            dest = (row.get("dest_pdf") or "").strip()
            pdf = data_dir / dest if dest else None
            if not pmid or not pdf or not pdf.exists():
                skipped += 1
                continue
            title = (row.get("title") or "").strip()
            year = (row.get("pub_year") or "").strip()
            try:
                text = extract_pdf_text(pdf)
            except Exception as e:
                logger.warning("PDF 读取失败 %s: %s", pdf.name, e)
                skipped += 1
                continue
            docs.append(
                EvidenceDoc(
                    doc_id=f"pmid:{pmid}",
                    source="pubmed",
                    title=title[:300] or pdf.stem,
                    text=text,
                    year=int(year) if year.isdigit() else None,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    doi="",
                    tags=["collection500", "fulltext"],
                    evidence_level=normalize_evidence_level("", title),
                    journal="",
                    extra={
                        "if_value": row.get("if_value", ""),  # 影响因子
                        "q_value": row.get("q_value", ""),
                        "nlm_unique_id": row.get("nlm_unique_id", ""),
                    },
                )
            )
            if len(docs) % 100 == 0:
                logger.info("已解析 %d 篇", len(docs))
    out = settings.raw_path / "collection_500.json"
    out.write_text(
        json.dumps([d.model_dump() for d in docs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("完成: %d 篇入库（跳过 %d）-> %s", len(docs), skipped, out)
    return len(docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量入库 500-collection PDF 语料")
    parser.add_argument("data_dir", type=Path, help="包含 PDF 的目录")
    parser.add_argument("manifest", type=Path, help="selected_manifest.csv 路径")
    args = parser.parse_args()
    ingest_collection(args.data_dir, args.manifest)


if __name__ == "__main__":
    main()
