from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/build_kb.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.ingest import (
    dedupe_by_doi_or_title,
    enrich_levels_from_text,
    export_ingest_report,
    load_docs,
    merge_docs,
    save_docs,
)
from src.ingest.clinicaltrials import ingest_clinicaltrials
from src.ingest.europepmc import ingest_europepmc
from src.ingest.local_docs import ingest_local
from src.ingest.pubmed import ingest_pubmed
from src.kb.chunking import docs_to_chunks, merge_tiny_chunks
from src.models import EvidenceDoc
from src.kb.store import (
    EvidenceStore,
    atomic_publish_chunks,
    export_store_stats,
    rebuild_collection_from_processed,
    validate_build_chunks,
)
from src.kb.wiki import generate_wiki_pages

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _run_ingest(*, skip_live: bool = False, retmax: int = 12) -> Path:
    settings = get_settings()
    settings.raw_path.mkdir(parents=True, exist_ok=True)
    local_docs = ingest_local(include_seed=True)
    save_docs(local_docs, settings.raw_path / "local_docs.json")
    if skip_live:
        # 离线模式：加载已保存的联网采集结果（避免覆盖队友/历史语料）
        pubmed_docs = load_docs(settings.raw_path / "pubmed.json")
        ct_docs = load_docs(settings.raw_path / "clinicaltrials.json")
        epmc_docs = load_docs(settings.raw_path / "europepmc.json")
        logger.info(
            "Offline ingest from saved raw: pubmed=%d clinicaltrials=%d europepmc=%d",
            len(pubmed_docs), len(ct_docs), len(epmc_docs),
        )
    else:
        pubmed_docs = ingest_pubmed(retmax_per_query=retmax)
        if pubmed_docs:
            save_docs(pubmed_docs, settings.raw_path / "pubmed.json")
        ct_docs = ingest_clinicaltrials(page_size=8)
        if ct_docs:
            save_docs(ct_docs, settings.raw_path / "clinicaltrials.json")
        epmc_docs = ingest_europepmc(page_size=10)
        if epmc_docs:
            save_docs(epmc_docs, settings.raw_path / "europepmc.json")
    # 人工整理的文献清单（如 D1PM 配套文献，doc_id=pmid:XXXX，可回查）
    lit_docs: list[EvidenceDoc] = []
    lit_path = settings.raw_path / "literature.json"
    if lit_path.exists():
        lit_docs = load_docs(lit_path)
        logger.info("Literature list -> %d docs", len(lit_docs))
    # 批量语料合集（如 500-collection，doc_id=pmid:XXXX + 影响因子）
    coll_docs: list[EvidenceDoc] = []
    coll_path = settings.raw_path / "collection_500.json"
    if coll_path.exists():
        coll_docs = load_docs(coll_path)
        logger.info("Collection corpus -> %d docs", len(coll_docs))
    # 主题精品知识库（如 salt_bp_kb.json，含 evidence_role 角色标记）
    kb_docs: list[EvidenceDoc] = []
    for kb_name in ("salt_bp_kb.json",):
        kb_path = settings.raw_path / kb_name
        if kb_path.exists():
            kb_docs += load_docs(kb_path)
    if kb_docs:
        logger.info("Curated topic KB -> %d docs", len(kb_docs))
    # 注意顺序：人工/精品语料（salt_bp 中文提炼+证据角色、literature 核实文献）
    # 必须排在批量采集之前——merge_docs 按 doc_id 先到先得，
    # 否则重叠 PMID 会被 pubmed/collection_500 的英文版本挤掉。
    all_docs = merge_docs(kb_docs, lit_docs, local_docs, pubmed_docs, ct_docs, epmc_docs, coll_docs)
    all_docs = dedupe_by_doi_or_title(all_docs)
    all_docs = enrich_levels_from_text(all_docs)
    out = settings.processed_path / "documents.json"
    save_docs(all_docs, out)
    report = export_ingest_report(all_docs, settings.processed_path / "ingest_report.md")
    logger.info("Merged %d documents -> %s", len(all_docs), out)
    logger.info("Ingest report -> %s", report)
    return out


def build_kb(
    *,
    skip_live: bool = False,
    skip_ingest: bool = False,
    reset: bool = True,
    incremental: bool = False,
    stats: bool = False,
) -> None:
    settings = get_settings()
    if not skip_ingest:
        _run_ingest(skip_live=skip_live)

    docs = load_docs(settings.processed_path / "documents.json")
    if not docs:
        logger.warning("No documents found; running local-only ingest")
        _run_ingest(skip_live=True)
        docs = load_docs(settings.processed_path / "documents.json")

    wiki_docs = generate_wiki_pages(docs)
    save_docs(wiki_docs, settings.processed_path / "wiki_docs.json")
    all_docs = merge_docs(docs, wiki_docs)
    save_docs(all_docs, settings.processed_path / "documents_with_wiki.json")

    if incremental:
        # 增量模式：指纹跳过未变 chunk + 剪枝已删除文档（与 kb_tools.py rebuild --incremental 同路径）
        n = rebuild_collection_from_processed(reset=False)
        logger.info("Incremental rebuild done: %d chunks upserted", n)
    else:
        chunks = docs_to_chunks(all_docs)
        chunks = merge_tiny_chunks(chunks)
        store = EvidenceStore()
        try:
            previous_count = store.count()
        except Exception:
            previous_count = None
        validate_build_chunks(chunks, previous_count=previous_count)
        n = atomic_publish_chunks(chunks, previous_count=previous_count)
        logger.info("Knowledge base ready: %d chunks (store count=%d)", n, store.count())
    if stats:
        export_store_stats(settings.processed_path / "store_stats.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence knowledge base")
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--incremental", action="store_true", help="增量重建：跳过未变、剪枝已删")
    parser.add_argument("--stats", action="store_true", help="构建后导出知识库统计")
    args = parser.parse_args()
    build_kb(
        skip_live=args.skip_live,
        skip_ingest=args.skip_ingest,
        reset=not args.no_reset,  # kept for CLI compat; publish path always uses atomic merge
        incremental=args.incremental,
        stats=args.stats,
    )


if __name__ == "__main__":
    main()
