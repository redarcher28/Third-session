from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import get_settings
from src.ingest import merge_docs, save_docs
from src.ingest.clinicaltrials import ingest_clinicaltrials
from src.ingest.europepmc import ingest_europepmc
from src.ingest.local_docs import ingest_local
from src.ingest.pubmed import ingest_pubmed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_ingest(
    *,
    skip_live: bool = False,
    retmax: int = 12,
) -> Path:
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

    all_docs = merge_docs(local_docs, pubmed_docs, ct_docs, epmc_docs)
    out = settings.processed_path / "documents.json"
    save_docs(all_docs, out)
    logger.info("Merged %d documents -> %s", len(all_docs), out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest evidence sources")
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Only use local seed docs (no PubMed/CT/EPMC network calls)",
    )
    parser.add_argument("--retmax", type=int, default=12)
    args = parser.parse_args()
    run_ingest(skip_live=args.skip_live, retmax=args.retmax)


if __name__ == "__main__":
    main()
