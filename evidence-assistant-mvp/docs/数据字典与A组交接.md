# Data dictionary and A-group handoff

## Shared contract

`EvidenceDoc` and `Chunk` carry `citation_eligible`, `record_type`, `source_locator`, and `provenance`. Chroma stores the first three directly and `provenance` as a JSON string because metadata values are scalar-only.

- `published_evidence`: citation eligible.
- `trial_registration`: `evidence_level=other`; cite registration facts only, never efficacy.
- `teaching_sample`, `wiki_demo`, `wiki_summary`: not citation eligible.
- `local_public_document`: eligibility depends on a verifiable local locator.

## B-group handoff

Use `data/processed/documents_with_wiki.json` for document lineage and parse Chroma `provenance` JSON for raw-response and dedupe lineage. Do not cite records where `citation_eligible=false`; do not treat trial registrations as efficacy evidence. Wiki is a navigation layer, and its supplemental evidence results must exclude `source=wiki`.

## Safe validation build

```powershell
$env:CHROMA_DIR = 'data/chroma-a-validation'
python scripts/build_kb.py --skip-live
```

`--reset` is explicit and runs only after traceability validation succeeds.
