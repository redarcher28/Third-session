"""采集质量报告的回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.ingest import export_ingest_report
from src.models import EvidenceDoc


class IngestReportTests(unittest.TestCase):
    def test_report_covers_traceability_duplicates_and_topic_quality(self) -> None:
        docs = [
            EvidenceDoc(
                doc_id="pmid:10001",
                source="pubmed",
                title="Sodium reduction guideline",
                text="Guideline summary",
                year=2024,
                url="https://pubmed.ncbi.nlm.nih.gov/10001/",
                tags=["salt_bp"],
                evidence_level="guideline",
                journal="Example Journal",
                doi="10.1000/example",
                extra={
                    "pmcid": "PMC10001",
                    "source_type": "guideline",
                    "evidence_role": "指南",
                },
            ),
            EvidenceDoc(
                doc_id="nct:NCT00000001",
                source="clinicaltrials",
                title="Sodium reduction trial",
                text="Trial summary",
                year=2023,
                url="https://clinicaltrials.gov/study/NCT00000001",
                tags=["salt_bp"],
                evidence_level="rct",
                extra={"nct": "NCT00000001"},
            ),
            EvidenceDoc(
                doc_id="local:duplicate-a",
                source="local",
                title="Duplicate candidate",
                text="Local note",
                tags=["local_topic"],
                evidence_level="ebook",
                doi="10.1000/duplicate",
            ),
            EvidenceDoc(
                doc_id="local:duplicate-b",
                source="local",
                title="Duplicate candidate",
                text="Another local note",
                tags=["local_topic"],
                evidence_level="ebook",
                doi="10.1000/duplicate",
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ingest_report.md"
            export_ingest_report(docs, output)
            report = output.read_text(encoding="utf-8")

        self.assertIn("- URL 覆盖率: 2/4 (50.0%)", report)
        self.assertIn("- PMID 标识: 1/4 (25.0%)", report)
        self.assertIn("- NCT 标识: 1/4 (25.0%)", report)
        self.assertIn("- PMCID 标识: 1/4 (25.0%)", report)
        self.assertIn("- evidence_role 字段覆盖率: 1/4 (25.0%)", report)
        self.assertIn("- 重复 DOI 组: 1 组，涉及 2 条文档", report)
        self.assertIn("| salt_bp | 2 | 1 | 0 | 1 | 2 |", report)
        self.assertIn("- 未发现同时满足“至少 5 条文档且无高等级证据”的标签。", report)


if __name__ == "__main__":
    unittest.main()
