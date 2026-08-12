# -*- coding: utf-8 -*-
"""证据充分性驱动的自适应最小证据集构建（规则版，不依赖 LLM/Dense）。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from src.kb.assertions import (
    CLINICAL_CONCLUSION_SCOPES,
    EvidenceAssertion,
    can_support_claim_scope,
    load_assertions,
    processed_assertions_path,
    production_assertions,
)

logger = logging.getLogger(__name__)

EvidenceStatus = Literal["sufficient", "partial", "insufficient", "unmapped"]

LEVEL_SCORE = {
    "guideline": 1.25,
    "meta": 1.2,
    "rct": 1.15,
    "observational": 1.05,
    "wiki": 0.9,
    "ebook": 0.85,
    "other": 0.8,
}

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "dash_diet": ("dash", "达什", "终止高血压膳食"),
    "sodium_reduction": ("限钠", "减钠", "少盐", "钠摄入", "sodium", "salt intake", "低钠"),
    "mediterranean_diet": ("地中海", "mediterranean"),
    "statin_therapy": ("他汀", "statin", "匹伐他汀", "pitavastatin"),
    "hypertension_long_term": ("长期吃药", "长期服药", "长期管理", "持续治疗", "随访管理"),
}

UNIT_SIGNAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "efficacy": ("是否有效", "有没有用", "能否降低", "证据是什么", "有什么证据", "有效吗", "降压", "降低"),
    "risk": ("风险", "副作用", "安全性", "不良"),
    "population": ("哪些人", "适用于谁", "适用人群", "什么人", "人群"),
    "long_term": ("长期", "持续", "管理", "依从"),
    "protocol": ("招募", "试验状态", "主要结局", "研究设计", "nct", "临床试验注册"),
}

UNIT_QUERY_HINTS = {
    "efficacy": "efficacy evidence outcome blood pressure",
    "risk": "safety adverse risk side effect",
    "population": "population adults eligible patients",
    "long_term": "long-term management follow-up adherence",
    "protocol": "clinical trial registry recruitment primary outcome",
}


@dataclass
class EvidenceUnit:
    unit_id: str
    required: bool
    claim_scope: str
    query_hint: str
    priority: int = 0


@dataclass
class CoverageEntry:
    unit_id: str
    assertion_id: str
    chunk_id: str
    support_strength: float
    evidence_level: str
    citation_eligible: bool
    record_type: str
    reason: str


@dataclass
class EvidencePlan:
    status: EvidenceStatus
    selected_contexts: list[dict[str, Any]] = field(default_factory=list)
    matched_assertion_ids: list[str] = field(default_factory=list)
    allowed_claims: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    blocked_claim_scopes: list[str] = field(default_factory=list)
    retrieval_rounds: int = 0
    selection_explanation: list[str] = field(default_factory=list)
    topic: str = ""
    units: list[EvidenceUnit] = field(default_factory=list)

    def to_retrieval_fields(self) -> dict[str, Any]:
        return {
            "evidence_status": self.status,
            "matched_assertion_ids": list(self.matched_assertion_ids),
            "allowed_claims": list(self.allowed_claims),
            "limitations": list(self.limitations),
            "missing_evidence": list(self.missing_evidence),
            "blocked_claim_scopes": list(self.blocked_claim_scopes),
            "selection_explanation": list(self.selection_explanation),
            "retrieval_rounds": self.retrieval_rounds,
            "assertion_topic": self.topic,
        }


def detect_topic(question: str) -> str | None:
    q = question.lower()
    for topic, keys in TOPIC_KEYWORDS.items():
        if any(k.lower() in q for k in keys):
            return topic
    return None


def identify_evidence_units(question: str, *, topic: str | None = None) -> list[EvidenceUnit]:
    """规则识别回答单元；不做 LLM 拆题。"""
    q = question.lower()
    units: list[EvidenceUnit] = []
    hit_ids: set[str] = set()

    for unit_id, patterns in UNIT_SIGNAL_PATTERNS.items():
        if any(p.lower() in q for p in patterns):
            hit_ids.add(unit_id)

    if not hit_ids:
        # 主题默认包：保证至少有主结论单元
        if topic == "hypertension_long_term":
            hit_ids = {"long_term", "efficacy"}
        elif topic:
            hit_ids = {"efficacy"}
        else:
            hit_ids = {"efficacy"}

    # protocol 问题以 protocol 为主，不强制疗效
    if "protocol" in hit_ids and any(
        p in q for p in ("招募", "nct", "临床试验注册", "试验状态", "主要结局")
    ):
        hit_ids = {"protocol"} | ({"population"} if "population" in hit_ids else set())

    priority = 0
    for unit_id in ("efficacy", "risk", "population", "long_term", "protocol"):
        if unit_id not in hit_ids:
            continue
        required = unit_id in {"efficacy", "protocol", "long_term"} or unit_id in hit_ids
        # 主结论：efficacy/protocol/long_term 为 required；risk/population 若被问到也为 required
        if unit_id in {"risk", "population"}:
            required = True
        units.append(
            EvidenceUnit(
                unit_id=unit_id,
                required=required,
                claim_scope="protocol" if unit_id == "protocol" else unit_id,
                query_hint=UNIT_QUERY_HINTS[unit_id],
                priority=priority,
            )
        )
        priority += 1
    return units


def _context_key(ctx: dict[str, Any]) -> str:
    return str(ctx.get("chunk_id") or ctx.get("doc_id") or id(ctx))


def match_assertions_for_topic(
    assertions: list[EvidenceAssertion],
    topic: str,
) -> list[EvidenceAssertion]:
    return [a for a in assertions if a.topic == topic]


def build_coverage_entries(
    units: list[EvidenceUnit],
    candidates: list[dict[str, Any]],
    assertions: list[EvidenceAssertion],
) -> list[CoverageEntry]:
    by_chunk = {_context_key(c): c for c in candidates}
    entries: list[CoverageEntry] = []
    for unit in units:
        for assertion in assertions:
            scope_match = assertion.claim_scope == unit.claim_scope
            if unit.unit_id == "protocol" and assertion.claim_scope in {
                "protocol",
                "recruitment_status",
                "primary_outcome",
            }:
                scope_match = True
            if not scope_match:
                continue
            for support in assertion.supports:
                ctx = by_chunk.get(support.chunk_id)
                if ctx is None:
                    # 也允许 doc_id 命中候选中的同文献块
                    ctx = next(
                        (c for c in candidates if str(c.get("doc_id")) == support.doc_id),
                        None,
                    )
                if ctx is None:
                    continue
                if not can_support_claim_scope(ctx, unit.claim_scope):
                    continue
                if not can_support_claim_scope(ctx, support.use):
                    continue
                level = str(ctx.get("evidence_level") or "other")
                strength = 1.0 * LEVEL_SCORE.get(level, 0.8)
                if assertion.review_status == "human_reviewed":
                    strength += 0.35
                else:
                    # 未审核卡不参与强覆盖（调用方应只传入 production 卡）
                    strength -= 0.5
                entries.append(
                    CoverageEntry(
                        unit_id=unit.unit_id,
                        assertion_id=assertion.assertion_id,
                        chunk_id=str(ctx.get("chunk_id") or support.chunk_id),
                        support_strength=strength,
                        evidence_level=level,
                        citation_eligible=bool(ctx.get("citation_eligible", True)),
                        record_type=str(ctx.get("record_type") or ""),
                        reason="assertion_support",
                    )
                )
    return entries


def _select_greedy(
    units: list[EvidenceUnit],
    candidates: list[dict[str, Any]],
    entries: list[CoverageEntry],
    assertions: list[EvidenceAssertion],
    *,
    max_evidence: int = 5,
) -> tuple[list[dict[str, Any]], list[str], list[str], set[str], list[str]]:
    """返回 selected, explanations, matched_assertion_ids, covered_units, allowed_claims."""
    required_ids = {u.unit_id for u in units if u.required}
    covered: set[str] = set()
    selected: list[dict[str, Any]] = []
    selected_docs: set[str] = set()
    explanations: list[str] = []
    matched_assertions: list[str] = []
    used_chunks: set[str] = set()

    prod = {a.assertion_id: a for a in production_assertions(assertions)}
    entry_by_chunk: dict[str, list[CoverageEntry]] = {}
    for e in entries:
        entry_by_chunk.setdefault(e.chunk_id, []).append(e)

    def marginal(ctx: dict[str, Any]) -> tuple[float, list[CoverageEntry]]:
        cid = _context_key(ctx)
        doc_id = str(ctx.get("doc_id") or "")
        related = entry_by_chunk.get(cid, [])
        # 若候选 chunk 不是主张卡绑定 id，尝试同 doc 的 entries
        if not related:
            related = [
                e
                for e in entries
                if any(
                    str(c.get("doc_id")) == doc_id and _context_key(c) == e.chunk_id
                    for c in candidates
                )
            ]
        new_units = {e.unit_id for e in related if e.unit_id not in covered}
        score = 0.0
        score += 2.5 * len(new_units & required_ids)
        score += 1.2 * len(new_units - required_ids)
        for e in related:
            score += 0.4 * e.support_strength
        level = str(ctx.get("evidence_level") or "other")
        score += LEVEL_SCORE.get(level, 0.8)
        if ctx.get("citation_eligible", True):
            score += 0.3
        if str(ctx.get("record_type")) == "trial_registry":
            # 只在 protocol 未覆盖时有价值
            if "protocol" in new_units:
                score += 0.5
            else:
                score -= 3.0
        if doc_id in selected_docs:
            score -= 2.5
        if cid in used_chunks:
            score -= 5.0
        # 无主张卡支持且试图撑临床结论
        if not related and any(u.claim_scope in CLINICAL_CONCLUSION_SCOPES for u in units):
            score -= 0.8
        return score, related

    while len(selected) < max_evidence:
        best = None
        best_score = 0.0
        best_related: list[CoverageEntry] = []
        for ctx in candidates:
            score, related = marginal(ctx)
            if score > best_score:
                best_score = score
                best = ctx
                best_related = related
        if best is None or best_score <= 0:
            break
        cid = _context_key(best)
        doc_id = str(best.get("doc_id") or "")
        selected.append(best)
        selected_docs.add(doc_id)
        used_chunks.add(cid)
        newly = {e.unit_id for e in best_related if e.unit_id not in covered}
        covered |= newly
        for e in best_related:
            if e.assertion_id not in matched_assertions and e.assertion_id in prod:
                matched_assertions.append(e.assertion_id)
        explanations.append(
            f"选择 {doc_id}：覆盖 {', '.join(sorted(newly)) or '无新单元'}；"
            f"level={best.get('evidence_level')}; eligible={best.get('citation_eligible', True)}"
        )
        if required_ids and required_ids.issubset(covered):
            break

    allowed_claims: list[str] = []
    for aid in matched_assertions:
        assertion = prod.get(aid)
        if assertion:
            allowed_claims.append(assertion.claim)

    # 冗余解释
    for ctx in candidates:
        doc_id = str(ctx.get("doc_id") or "")
        if doc_id in selected_docs and _context_key(ctx) not in used_chunks:
            explanations.append(f"未选择 {doc_id} 的其他切块：同文献冗余")
            break

    return selected, explanations, matched_assertions, covered, allowed_claims


def decide_status(
    units: list[EvidenceUnit],
    covered: set[str],
    allowed_claims: list[str],
    *,
    topic: str | None,
    has_production_assertions: bool,
) -> tuple[EvidenceStatus, list[str]]:
    if not topic:
        return "unmapped", ["当前问题未进入主张卡充分性判定（主题未映射）"]
    if not has_production_assertions:
        # 仅有 needs_human_review 草稿时：不宣称 sufficient，也不阻断既有问答；
        # 回落到 unmapped，由流水线保留 legacy 检索生成。
        return (
            "unmapped",
            ["主题相关主张卡尚未 human_reviewed，未进入充分性判定"],
        )

    required = [u for u in units if u.required]
    missing = [u.unit_id for u in required if u.unit_id not in covered]
    primary_ids = {u.unit_id for u in required if u.unit_id in {"efficacy", "protocol", "long_term"}}
    primary_missing = [u for u in primary_ids if u not in covered]

    if not allowed_claims or primary_missing:
        reasons = [f"未覆盖主结论单元：{u}" for u in primary_missing] or [
            "主结论缺少已审核主张卡支持"
        ]
        return "insufficient", reasons
    if missing:
        return "partial", [f"未覆盖次要/必需单元：{u}" for u in missing]
    return "sufficient", []


RetrieveFn = Callable[[str, int], list[dict[str, Any]]]


def build_evidence_plan(
    question: str,
    *,
    initial_candidates: list[dict[str, Any]],
    retrieve_fn: RetrieveFn | None = None,
    assertions: list[EvidenceAssertion] | None = None,
    max_evidence: int = 5,
    max_rounds: int = 2,
) -> EvidencePlan:
    """
    构建最小充分证据集计划。

    assertions 为 None 时尝试加载 processed/curated；皆空则 unmapped/disabled 行为。
    """
    if assertions is None:
        processed = processed_assertions_path()
        curated = None
        try:
            from src.kb.assertions import curated_assertions_path

            curated = curated_assertions_path()
        except Exception:
            curated = None
        if processed.exists():
            assertions = load_assertions(processed)
        elif curated and curated.exists():
            assertions = load_assertions(curated)
        else:
            assertions = []

    topic = detect_topic(question)
    blocked = ["dosage", "individual_treatment"]
    if not assertions:
        return EvidencePlan(
            status="unmapped",
            selected_contexts=list(initial_candidates)[:max_evidence],
            missing_evidence=["主张卡功能不可用或未配置"],
            blocked_claim_scopes=blocked,
            selection_explanation=["assertions unavailable; fallback to legacy top-k"],
            retrieval_rounds=0,
        )

    if not topic:
        return EvidencePlan(
            status="unmapped",
            selected_contexts=list(initial_candidates)[:max_evidence],
            missing_evidence=["问题主题未映射到主张卡覆盖范围"],
            blocked_claim_scopes=blocked,
            selection_explanation=["topic unmapped; not marked sufficient"],
            retrieval_rounds=0,
        )

    units = identify_evidence_units(question, topic=topic)
    topic_assertions = match_assertions_for_topic(assertions, topic)
    prod = production_assertions(topic_assertions)
    if not prod:
        return EvidencePlan(
            status="unmapped",
            selected_contexts=list(initial_candidates)[:max_evidence],
            missing_evidence=["主题相关主张卡尚未 human_reviewed，未进入充分性判定"],
            blocked_claim_scopes=blocked,
            selection_explanation=["no human_reviewed assertions for topic; legacy top-k"],
            retrieval_rounds=0,
            topic=topic,
            units=units,
        )

    candidates = list(initial_candidates)
    rounds = 0
    selected: list[dict[str, Any]] = []
    explanations: list[str] = []
    matched_ids: list[str] = []
    covered: set[str] = set()
    allowed_claims: list[str] = []

    while True:
        entries = build_coverage_entries(units, candidates, prod)
        selected, explanations, matched_ids, covered, allowed_claims = _select_greedy(
            units,
            candidates,
            entries,
            prod,
            max_evidence=max_evidence,
        )
        status, missing = decide_status(
            units,
            covered,
            allowed_claims,
            topic=topic,
            has_production_assertions=bool(prod),
        )
        if status in {"sufficient", "unmapped"}:
            break
        if rounds >= max_rounds or retrieve_fn is None:
            break
        # 定向补检：针对缺失单元
        missing_units = [u for u in units if u.required and u.unit_id not in covered]
        if not missing_units:
            break
        hint = " ".join(u.query_hint for u in missing_units[:2])
        query = f"{question} {topic.replace('_', ' ')} {hint}"
        extra = retrieve_fn(query, 16)
        rounds += 1
        seen = {_context_key(c) for c in candidates}
        for item in extra:
            key = _context_key(item)
            if key not in seen:
                candidates.append(item)
                seen.add(key)
        explanations.append(f"定向补检第 {rounds} 轮：query_hint={hint}")

    limitations: list[str] = []
    for aid in matched_ids:
        for a in prod:
            if a.assertion_id == aid:
                limitations.extend(a.limitations)

    # 去重 limitations
    dedup_lim: list[str] = []
    for lim in limitations:
        if lim not in dedup_lim:
            dedup_lim.append(lim)

    status, missing = decide_status(
        units,
        covered,
        allowed_claims,
        topic=topic,
        has_production_assertions=bool(prod),
    )

    if status == "insufficient" and not selected:
        selected = list(initial_candidates)[: max(1, min(3, max_evidence))]

    return EvidencePlan(
        status=status,
        selected_contexts=selected,
        matched_assertion_ids=matched_ids,
        allowed_claims=allowed_claims,
        limitations=dedup_lim,
        missing_evidence=missing,
        blocked_claim_scopes=blocked,
        retrieval_rounds=rounds,
        selection_explanation=explanations,
        topic=topic or "",
        units=units,
    )


def render_allowed_claims_answer(
    plan: EvidencePlan,
    citations_block: str = "",
) -> str:
    """确定性证据卡式回答模板（不依赖模型自由发挥）。"""
    lines: list[str] = []
    if plan.status == "insufficient":
        lines.append("当前缺少可支持主结论的已审核证据主张，无法给出明确临床结论。")
    elif plan.status == "partial":
        lines.append("以下结论仅覆盖已获支持的部分；其余要点证据不足，已限制生成范围。")
    else:
        lines.append("基于已审核主张卡与选中证据，可支持如下结论：")

    if plan.allowed_claims:
        for i, claim in enumerate(plan.allowed_claims, start=1):
            cite = f" [{i}]" if citations_block else ""
            lines.append(f"- {claim}{cite}")
    else:
        lines.append("- （无允许生成的已审核主张）")

    if plan.limitations:
        lines.append("局限：")
        for lim in plan.limitations[:6]:
            lines.append(f"- {lim}")
    if plan.missing_evidence:
        lines.append("证据缺口：")
        for miss in plan.missing_evidence:
            lines.append(f"- {miss}")
    if citations_block:
        lines.append(citations_block)
    return "\n".join(lines)


def question_mentions_protocol_only(question: str) -> bool:
    q = question.lower()
    return bool(re.search(r"nct\d{8}|临床试验注册|招募状态|主要结局|研究设计", q))
