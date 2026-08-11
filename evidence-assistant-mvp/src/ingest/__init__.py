from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src import PROJECT_ROOT
from src.models import EvidenceDoc, EvidenceLevel

@dataclass
class IngestRun:
    run_id: str
    source: str
    queries: list[str]
    request_params: dict
    started_at: str
    directory: Path
    raw_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def save_response(self, content: str | bytes, suffix: str) -> str:
        path = self.directory / f"response-{len(self.raw_files) + 1:03d}.{suffix.lstrip('.')}"
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        self.raw_files.append(path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix())
        return self.raw_files[-1]

    def finalize(self, normalized_docs: list[EvidenceDoc]) -> Path:
        manifest = {"run_id": self.run_id, "source": self.source, "queries": self.queries,
                    "request_params": self.request_params, "started_at": self.started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(), "raw_files": self.raw_files,
                    "raw_response_count": len(self.raw_files), "normalized_document_count": len(normalized_docs),
                    "errors": self.errors}
        path = self.directory / "manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

def start_ingest_run(source: str, queries: list[str], request_params: dict) -> IngestRun:
    from src.config import get_settings
    now = datetime.now(timezone.utc)
    run_id = f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    directory = get_settings().raw_path / source / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return IngestRun(run_id, source, list(queries), dict(request_params), now.isoformat(), directory)

def save_docs(docs: Iterable[EvidenceDoc], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [d.model_dump() for d in docs]
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(items)

def load_docs(path: Path) -> list[EvidenceDoc]:
    if not path.exists(): return []
    return [EvidenceDoc.model_validate(x) for x in json.loads(path.read_text(encoding="utf-8"))]

def merge_docs(*doc_lists: list[EvidenceDoc]) -> list[EvidenceDoc]:
    seen, out = set(), []
    for docs in doc_lists:
        for doc in docs:
            if doc.doc_id not in seen:
                seen.add(doc.doc_id); out.append(doc)
    return out

def normalize_evidence_level(raw_type: str, title: str) -> EvidenceLevel:
    blob = f"{raw_type} {title}".lower()
    if any(k in blob for k in ("guideline", "practice guideline", "consensus")): return "guideline"
    if any(k in blob for k in ("meta-analysis", "meta analysis", "systematic review")): return "meta"
    if any(k in blob for k in ("randomized", "randomised", "clinical trial", "controlled trial")): return "rct"
    if any(k in blob for k in ("cohort", "observational", "case-control", "cross-sectional", "case series", "case report")): return "observational"
    if any(k in blob for k in ("book", "ebook", "chapter")): return "ebook"
    if "wiki" in blob: return "wiki"
    return "other"

def _norm_doi(doi: str) -> str:
    value = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.startswith(prefix): value = value[len(prefix):]
    return value.strip()

def _norm_title(title: str) -> str:
    return re.sub(r"[^\\w\\u4e00-\\u9fff]+", "", re.sub(r"\\s+", " ", title.lower()).strip())

def _completeness(doc: EvidenceDoc) -> int:
    return (len((doc.text or "").strip()) + 1000 * bool(doc.year) + 500 * bool(doc.url) +
            500 * bool(doc.journal) + 300 * bool(doc.doi) + 100 * len(doc.tags))

def dedupe_with_stats(docs: list[EvidenceDoc]) -> tuple[list[EvidenceDoc], dict]:
    parent = list(range(len(docs))); links: list[tuple[int, int, str]] = []
    def find(i: int) -> int:
        while parent[i] != i: parent[i], i = parent[parent[i]], parent[i]
        return i
    def join(a: int, b: int, reason: str) -> None:
        left, right = find(a), find(b)
        if left != right: parent[right] = left
        links.append((a, b, reason))
    by_doi: dict[str, int] = {}; by_title: dict[str, int] = {}
    for i, doc in enumerate(docs):
        if doc.record_type == "trial_registration": continue
        doi, title = _norm_doi(doc.doi), _norm_title(doc.title)
        if doi:
            if doi in by_doi: join(by_doi[doi], i, "normalized_doi")
            else: by_doi[doi] = i
        if title:
            if title in by_title: join(by_title[title], i, "normalized_title")
            else: by_title[title] = i
    groups: dict[int, list[int]] = {}
    for i in range(len(docs)): groups.setdefault(find(i), []).append(i)
    result: list[EvidenceDoc] = []; reasons: Counter[str] = Counter()
    for indices in sorted(groups.values(), key=min):
        best_i = max(indices, key=lambda i: _completeness(docs[i])); best = docs[best_i].model_copy(deep=True)
        if len(indices) > 1:
            records = list(best.provenance.get("merged_records", []))
            for i in indices:
                if i == best_i: continue
                why = sorted({reason for a,b,reason in links if i in (a,b) or best_i in (a,b)}) or ["transitive_doi_or_title"]
                reasons.update(why); other = docs[i]
                records.append({"source": other.source, "doc_id": other.doc_id, "dedupe_reason": why, "provenance": other.provenance})
            best.provenance["merged_records"] = records
        result.append(best)
    return result, {"before_count": len(docs), "after_count": len(result), "deduped_count": len(docs)-len(result), "reasons": dict(reasons)}

def dedupe_by_doi_or_title(docs: list[EvidenceDoc]) -> list[EvidenceDoc]:
    return dedupe_with_stats(docs)[0]

def export_ingest_report(docs: list[EvidenceDoc], out_path: Path, *, dedupe_stats: dict | None = None) -> Path:
    total = len(docs); sources = Counter(d.source for d in docs); levels = Counter(d.evidence_level for d in docs)
    record_types = Counter(d.record_type for d in docs); years = Counter(d.year for d in docs)
    stats = dedupe_stats or {"before_count": total, "deduped_count": 0, "reasons": {}}
    def section(name: str, counter: Counter) -> list[str]:
        return ["", f"## {name}", ""] + [f"- {k if k is not None else 'unknown'}: {v}" for k,v in sorted(counter.items(), key=lambda x: str(x[0]))]
    lines = ["# Ingest quality report", "", f"- generated_at: {datetime.now():%Y-%m-%d %H:%M}", f"- documents_total: {total}",
             f"- before_dedupe: {stats['before_count']}", f"- deduped: {stats['deduped_count']}",
             f"- missing_url_or_source_locator: {sum(not (d.url or d.source_locator) for d in docs)}",
             f"- missing_abstract: {sum(not (d.text or '').strip() for d in docs)}", f"- missing_identifier: {sum(not (d.doi or d.doc_id) for d in docs)}",
             f"- teaching_samples: {sum(d.record_type == 'teaching_sample' for d in docs)}",
             f"- clinicaltrials_registrations: {sum(d.record_type == 'trial_registration' for d in docs)}",
             f"- citation_eligible_false: {sum(not d.citation_eligible for d in docs)}"]
    lines += section("Source distribution", sources) + section("Evidence level distribution", levels) + section("Record type distribution", record_types) + section("Year distribution", years)
    lines += ["", "## Dedupe reasons", ""] + ([f"- {k}: {v}" for k,v in stats["reasons"].items()] or ["- none"])
    out_path.parent.mkdir(parents=True, exist_ok=True); out_path.write_text("\\n".join(lines)+"\\n", encoding="utf-8")
    return out_path
