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
from src.ingest import dedupe_by_doi_or_title, export_ingest_report, load_docs, merge_docs, save_docs
from src.ingest.clinicaltrials import ingest_clinicaltrials
from src.ingest.europepmc import ingest_europepmc
from src.ingest.local_docs import ingest_local
from src.ingest.pubmed import ingest_pubmed
from src.kb.chunking import docs_to_chunks, merge_tiny_chunks, validate_chunk_traceability
from src.models import EvidenceDoc
from src.kb.store import EvidenceStore, export_store_stats, rebuild_collection_from_processed
from src.kb.wiki import generate_wiki_pages

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _run_ingest(*, skip_live: bool = False, retmax: int = 12) -> Path:
    settings = get_settings()
    settings.raw_path.mkdir(parents=True, exist_ok=True)
    local_docs = ingest_local(include_seed=True)
    save_docs(local_docs, settings.raw_path / "local_docs.json")
    pubmed_docs = [] if skip_live else ingest_pubmed(retmax_per_query=retmax)
    if pubmed_docs:
        save_docs(pubmed_docs, settings.raw_path / "pubmed.json")
    ct_docs = [] if skip_live else ingest_clinicaltrials(page_size=8)
    if ct_docs:
        save_docs(ct_docs, settings.raw_path / "clinicaltrials.json")
    epmc_docs = [] if skip_live else ingest_europepmc(page_size=10)
    if epmc_docs:
        save_docs(epmc_docs, settings.raw_path / "europepmc.json")
    # 人工整理的文献清单（如 D1PM 配套文献，doc_id=pmid:XXXX，可回查）
    lit_docs: list[EvidenceDoc] = []
    lit_path = settings.raw_path / "literature.json"
    if lit_path.exists():
        lit_docs = load_docs(lit_path)
        logger.info("Literature list -> %d docs", len(lit_docs))
    all_docs = merge_docs(local_docs, pubmed_docs, ct_docs, epmc_docs, lit_docs)
    all_docs = dedupe_by_doi_or_title(all_docs)
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
        trace = validate_chunk_traceability(chunks)
        if not trace["ok"]:
            logger.warning("Chunk traceability issues: %s", trace)
        store = EvidenceStore()
        if reset:
            store.reset()
        n = store.upsert_chunks(chunks)
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
        reset=not args.no_reset,
        incremental=args.incremental,
        stats=args.stats,
    )


if __name__ == "__main__":
    main()
