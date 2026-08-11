# -*- coding: utf-8 -*-
"""
把 JSONL 精品知识库（如「限钠与血压_知识库.jsonl」）转为 EvidenceDoc。

输出: data/raw/salt_bp_kb.json，build_kb 会自动并入采集流程。

字段映射：
    id → doc_id（pmid:xxxxx，不可改名）
    source_type → evidence_level（guideline/meta/rct/observational）
    evidence_role（指南/总览/因果/边界）→ tags + extra
    text（中文提炼摘要）→ 检索主字段；abstract_en 保留在 extra 供核对

用法:
    python scripts/ingest_jsonl_kb.py <xxx.jsonl> [--tag salt_bp]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/ingest_jsonl_kb.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings  # noqa: E402
from src.models import EvidenceDoc  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# source_type → 统一证据等级（与 src/models.py 枚举一致）
_SOURCE_TYPE_TO_LEVEL = {
    "guideline": "guideline",
    "umbrella_review": "meta",
    "systematic_review": "meta",
    "meta_analysis": "meta",
    "randomized_controlled_trial": "rct",
    "cohort_study": "observational",
}

_KEEP_EXTRA = ("source_type", "evidence_role", "authors", "pmcid", "nct", "abstract_en")


def ingest_jsonl_kb(jsonl_path: Path, tag: str) -> int:
    """把 JSONL 记录转为 EvidenceDoc 并保存。"""
    settings = get_settings()
    settings.raw_path.mkdir(parents=True, exist_ok=True)
    docs: list[EvidenceDoc] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        doc_id = str(rec.get("id") or "").strip()
        if not doc_id:
            logger.warning("跳过无 id 记录: %s", line[:80])
            continue
        extra = {k: rec.get(k) for k in _KEEP_EXTRA if rec.get(k) is not None}
        tags = [tag]
        if rec.get("evidence_role"):
            tags.append(str(rec["evidence_role"]))
        if rec.get("source_type"):
            tags.append(str(rec["source_type"]))
        docs.append(
            EvidenceDoc(
                doc_id=doc_id,
                source="pubmed",
                title=str(rec.get("title") or doc_id),
                text=str(rec.get("text") or ""),
                year=rec.get("year"),
                url=str(rec.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{doc_id.split(':')[-1]}/"),
                doi=str(rec.get("doi") or ""),
                journal=str(rec.get("journal") or ""),
                tags=tags,
                evidence_level=_SOURCE_TYPE_TO_LEVEL.get(str(rec.get("source_type") or ""), "other"),
                extra=extra,
            )
        )
    out = settings.raw_path / f"{tag}_kb.json"
    out.write_text(
        json.dumps([d.model_dump() for d in docs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("JSONL 知识库入库 %d 条 -> %s", len(docs), out)
    return len(docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="JSONL 精品知识库入库")
    parser.add_argument("jsonl", type=Path, help="JSONL 文件路径")
    parser.add_argument("--tag", default="salt_bp", help="语料标签（默认 salt_bp）")
    args = parser.parse_args()
    n = ingest_jsonl_kb(args.jsonl, args.tag)
    print(f"完成: {n} 条入库")


if __name__ == "__main__":
    main()
