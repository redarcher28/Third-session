# -*- coding: utf-8 -*-
"""可追溯证据主张卡：加载、hash、建库校验与 claim_scope 约束。"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from src import PROJECT_ROOT
from src.models import Chunk

logger = logging.getLogger(__name__)

ClaimScope = Literal[
    "efficacy",
    "risk",
    "population",
    "long_term",
    "protocol",
    "recruitment_status",
    "primary_outcome",
    "limitation",
]

ReviewStatus = Literal["needs_human_review", "human_reviewed", "rejected"]

CLINICAL_CONCLUSION_SCOPES = frozenset({"efficacy", "risk", "population", "long_term"})
PROTOCOL_SCOPES = frozenset({"protocol", "recruitment_status", "primary_outcome"})


class AssertionSupport(BaseModel):
    doc_id: str
    chunk_id: str
    quote_hash: str
    use: ClaimScope


class EvidenceAssertion(BaseModel):
    assertion_id: str
    topic: str
    claim: str
    claim_scope: ClaimScope
    population: str = ""
    intervention: str = ""
    outcome: str = ""
    limitations: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = "needs_human_review"
    reviewed_at: str | None = None
    supports: list[AssertionSupport] = Field(default_factory=list)


class AssertionValidationError(ValueError):
    """主张卡未通过硬校验，禁止作为构建产物发布。"""


def normalize_quote_text(text: str) -> str:
    return (text or "").strip()


def compute_quote_hash(text: str) -> str:
    digest = hashlib.sha256(normalize_quote_text(text).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def curated_assertions_path() -> Path:
    return PROJECT_ROOT / "data" / "curated" / "assertions.jsonl"


def processed_assertions_path(processed_dir: Path | None = None) -> Path:
    if processed_dir is None:
        from src.config import get_settings

        processed_dir = get_settings().processed_path
    return Path(processed_dir) / "assertions.jsonl"


def load_assertions(path: Path | None = None) -> list[EvidenceAssertion]:
    """加载主张卡；文件不存在时返回空列表（功能不可用，不伪造）。"""
    path = path or curated_assertions_path()
    if not path.exists():
        return []
    out: list[EvidenceAssertion] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(EvidenceAssertion.model_validate(json.loads(line)))
        except Exception as exc:
            raise AssertionValidationError(
                f"invalid assertion at {path}:{line_no}: {exc}"
            ) from exc
    return out


def production_assertions(assertions: Iterable[EvidenceAssertion]) -> list[EvidenceAssertion]:
    """仅 human_reviewed 可作为生产充分性/强结论锚点。"""
    return [a for a in assertions if a.review_status == "human_reviewed"]


def can_support_claim_scope(context: dict[str, Any], claim_scope: str) -> bool:
    """
    判断单条证据上下文是否允许支撑给定 claim_scope。

    试验注册可支撑 protocol 类事实；不得支撑疗效/风险/人群/长期管理结论。
    """
    record_type = str(context.get("record_type") or "")
    source = str(context.get("source") or "")
    eligible = bool(context.get("citation_eligible", True))
    scope = str(claim_scope or "")

    is_registry = record_type == "trial_registry" or source == "clinicaltrials"
    if scope in CLINICAL_CONCLUSION_SCOPES:
        if is_registry:
            return False
        if not eligible:
            return False
        if record_type in {"wiki_page", "local_doc"} and not eligible:
            return False
        return True
    if scope in PROTOCOL_SCOPES:
        return is_registry or eligible
    if scope == "limitation":
        return True
    return eligible and not is_registry


def _chunk_index(chunks: Iterable[Chunk]) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}


def validate_assertions_against_chunks(
    assertions: list[EvidenceAssertion],
    chunks: list[Chunk],
) -> dict[str, Any]:
    """
    建库前硬校验主张卡与待发布 Chunk 的一致性。

    返回:
        dict: ok, errors[], assertion_count, production_count
    """
    errors: list[dict[str, Any]] = []
    by_id = _chunk_index(chunks)
    seen_assertion_ids: set[str] = set()

    for assertion in assertions:
        aid = assertion.assertion_id
        if aid in seen_assertion_ids:
            errors.append(
                {
                    "assertion_id": aid,
                    "reason": "duplicate_assertion_id",
                    "doc_id": "",
                    "chunk_id": "",
                }
            )
            continue
        seen_assertion_ids.add(aid)

        if not assertion.topic.strip() or not assertion.claim.strip():
            errors.append(
                {
                    "assertion_id": aid,
                    "reason": "missing_topic_or_claim",
                    "doc_id": "",
                    "chunk_id": "",
                }
            )
        if not assertion.limitations:
            errors.append(
                {
                    "assertion_id": aid,
                    "reason": "missing_limitations",
                    "doc_id": "",
                    "chunk_id": "",
                }
            )
        if not assertion.supports:
            errors.append(
                {
                    "assertion_id": aid,
                    "reason": "missing_supports",
                    "doc_id": "",
                    "chunk_id": "",
                }
            )

        for support in assertion.supports:
            chunk = by_id.get(support.chunk_id)
            if chunk is None:
                errors.append(
                    {
                        "assertion_id": aid,
                        "reason": "chunk_not_found",
                        "doc_id": support.doc_id,
                        "chunk_id": support.chunk_id,
                    }
                )
                continue
            if chunk.doc_id != support.doc_id:
                errors.append(
                    {
                        "assertion_id": aid,
                        "reason": "doc_id_mismatch",
                        "doc_id": support.doc_id,
                        "chunk_id": support.chunk_id,
                    }
                )
            expected_hash = compute_quote_hash(chunk.text)
            if support.quote_hash != expected_hash:
                errors.append(
                    {
                        "assertion_id": aid,
                        "reason": "quote_hash_mismatch",
                        "doc_id": support.doc_id,
                        "chunk_id": support.chunk_id,
                    }
                )

            ctx = {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "record_type": chunk.record_type,
                "citation_eligible": chunk.citation_eligible,
                "url": chunk.url,
                "source_locator": chunk.source_locator,
            }
            # support.use 与主张 claim_scope 的临床约束都要检查
            for scope in {assertion.claim_scope, support.use}:
                if not can_support_claim_scope(ctx, scope):
                    errors.append(
                        {
                            "assertion_id": aid,
                            "reason": f"scope_not_allowed:{scope}",
                            "doc_id": support.doc_id,
                            "chunk_id": support.chunk_id,
                        }
                    )

            locator = (chunk.source_locator or chunk.url or chunk.doc_id or "").strip()
            if not locator:
                errors.append(
                    {
                        "assertion_id": aid,
                        "reason": "missing_locator",
                        "doc_id": support.doc_id,
                        "chunk_id": support.chunk_id,
                    }
                )

    report = {
        "ok": not errors,
        "errors": errors,
        "assertion_count": len(assertions),
        "production_count": len(production_assertions(assertions)),
    }
    return report


def validate_and_export_assertions(
    chunks: list[Chunk],
    *,
    curated_path: Path | None = None,
    processed_dir: Path | None = None,
) -> dict[str, Any]:
    """
    校验 curated 主张卡；成功则复制到 processed。

    curated 不存在时返回 disabled，不阻断建库。
    curated 存在但校验失败时抛出 AssertionValidationError。
    """
    curated_path = curated_path or curated_assertions_path()
    if not curated_path.exists():
        logger.info("Assertion cards unavailable (no %s); feature disabled", curated_path)
        return {
            "ok": True,
            "disabled": True,
            "errors": [],
            "assertion_count": 0,
            "production_count": 0,
        }

    assertions = load_assertions(curated_path)
    report = validate_assertions_against_chunks(assertions, chunks)
    if not report["ok"]:
        raise AssertionValidationError(
            f"assertion validation failed: {json.dumps(report['errors'][:20], ensure_ascii=False)}"
        )

    out_dir = processed_dir
    if out_dir is None:
        from src.config import get_settings

        out_dir = get_settings().processed_path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "assertions.jsonl"
    shutil.copy2(curated_path, dest)
    report["disabled"] = False
    report["exported_to"] = str(dest)
    logger.info(
        "Assertion cards validated: total=%d production=%d -> %s",
        report["assertion_count"],
        report["production_count"],
        dest,
    )
    return report
