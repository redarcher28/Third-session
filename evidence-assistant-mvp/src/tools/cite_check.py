# -*- coding: utf-8 -*-
"""
引用校验工具：检测假引用、无效编号、编造 PMID/NCT。
"""

from __future__ import annotations

import re
from typing import Any

from src.generation.answer import extract_citation_indices


PMID_RE = re.compile(r"PMID[:\s]*([0-9]{5,9})(?![0-9])", re.I)
NCT_RE = re.compile(r"(NCT\d{8})(?![0-9])", re.I)
DOC_RE = re.compile(r"((?:pmid|nct|epmc|local|wiki):[^\s\]，。；;,]+)", re.I)


def verify_citations(answer: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    校验回答中的引用是否都能在当次证据中找到。

    参数:
        answer: 模型回答文本。
        contexts: 当次检索证据列表。

    返回:
        dict: 校验明细，关键字段包括：
            - ok (bool): 是否全部合法
            - used_brackets (list[int]): 使用的 [n]
            - invalid_brackets (list[int]): 越界编号
            - fake_pmids / fake_ncts / fake_docs: 无法核实的标识
            - has_citations (bool): 是否出现引用
    """
    allowed_idx = set(range(1, len(contexts) + 1))
    used = extract_citation_indices(answer)
    invalid_brackets = [i for i in used if i not in allowed_idx]

    context_ids: set[str] = set()
    context_pmids: set[str] = set()
    context_ncts: set[str] = set()
    for c in contexts:
        doc_id = str(c.get("doc_id") or "")
        context_ids.add(doc_id.lower())
        if doc_id.lower().startswith("pmid:"):
            context_pmids.add(doc_id.split(":", 1)[1])
        if doc_id.lower().startswith("nct:"):
            context_ncts.add(doc_id.split(":", 1)[1].upper())
        for m in PMID_RE.findall(str(c.get("url") or "") + " " + doc_id):
            context_pmids.add(m)
        for m in NCT_RE.findall(str(c.get("url") or "") + " " + doc_id):
            context_ncts.add(m.upper())

    claimed_pmids = PMID_RE.findall(answer)
    claimed_ncts = [m.upper() for m in NCT_RE.findall(answer)]
    claimed_docs = [m.lower() for m in DOC_RE.findall(answer)]

    fake_pmids = [p for p in claimed_pmids if p not in context_pmids]
    fake_ncts = [n for n in claimed_ncts if n not in context_ncts]
    fake_docs = [d for d in claimed_docs if d not in context_ids]

    invent_phrases = 0
    for pat in [
        r"根据某项研究",
        r"著名的 .+ 试验",
        r"发表于《[^》]+》",
        r"et al\.",
    ]:
        invent_phrases += len(re.findall(pat, answer))

    ok = not invalid_brackets and not fake_pmids and not fake_ncts and not fake_docs
    return {
        "ok": ok,
        "used_brackets": used,
        "invalid_brackets": invalid_brackets,
        "claimed_pmids": claimed_pmids,
        "fake_pmids": fake_pmids,
        "claimed_ncts": claimed_ncts,
        "fake_ncts": fake_ncts,
        "fake_docs": fake_docs,
        "has_citations": bool(used),
        "invent_phrase_hits": invent_phrases,
    }


def strip_invalid_claims(answer: str, check: dict[str, Any]) -> str:
    """
    校验失败时，在回答末尾追加「无法核实」提示（不直接删除全文）。

    参数:
        answer: 原回答。
        check: verify_citations 返回值。

    返回:
        str: 可能追加提示后的回答。
    """
    if check.get("ok", True):
        return answer
    notes = []
    if check.get("invalid_brackets"):
        notes.append(f"无效引用编号: {check['invalid_brackets']}")
    if check.get("fake_pmids"):
        notes.append(f"无法核实的 PMID: {check['fake_pmids']}")
    if check.get("fake_ncts"):
        notes.append(f"无法核实的 NCT: {check['fake_ncts']}")
    if check.get("fake_docs"):
        notes.append(f"无法核实的文档 ID: {check['fake_docs']}")
    note = "；".join(notes) if notes else "检测到可疑引用"
    return (
        answer.rstrip()
        + f"\n\n> 引用校验：{note}。请仅采信与证据面板一致的引用。"
    )


def repair_answer_with_valid_cites(
    answer: str,
    contexts: list[dict[str, Any]],
    check: dict[str, Any],
) -> str:
    """
    当引用校验失败时，基于合法证据修复回答中的引用。

    参数:
        answer: 原始生成回答。
        contexts: 当次检索证据。
        check: verify_citations 的返回结果。

    返回:
        str: 修复后的回答文本。

    作用:
        降低假引用残留，提升赛道三「假引用减少」指标表现。
    """
    if not check or check.get("ok", True):
        return answer

    n_contexts = len(contexts)
    valid = set(range(1, n_contexts + 1))

    # 1) 删除越界编号 [n]
    def _drop_bad(match: "re.Match[str]") -> str:
        num = int(match.group(1))
        return f"[{num}]" if num in valid else ""

    fixed = re.sub(r"\[(\d+)\]", _drop_bad, answer)

    # 2) 删除无法核实的 PMID / NCT / 文档 ID（保留原文文字，只去掉标识符）
    fake_pmids = check.get("fake_pmids") or []
    fake_ncts = check.get("fake_ncts") or []
    fake_docs = check.get("fake_docs") or []
    for pmid in fake_pmids:
        fixed = re.sub(
            rf"PMID[:\s]*{re.escape(str(pmid))}(?![0-9])",
            "PMID[未核实]",
            fixed,
            flags=re.I,
        )
    for nct in fake_ncts:
        fixed = re.sub(
            rf"{re.escape(str(nct))}(?![0-9])",
            "[试验编号未核实]",
            fixed,
            flags=re.I,
        )
    for doc in fake_docs:
        fixed = re.sub(
            rf"{re.escape(str(doc))}(?![0-9A-Za-z])",
            "[文献ID未核实]",
            fixed,
            flags=re.I,
        )

    # 3) 若引用被清空而证据存在，为正文首段补一个有效引用
    if not re.search(r"\[\d+\]", fixed) and n_contexts > 0:
        fixed = fixed.rstrip()
        if not fixed.endswith(("[1]", "。", ".")):
            fixed += "[1]"

    # 4) 在线模式且仍有可疑引用时，用 LLM 结构化重写一次；失败回退规则式结果
    suspicious = bool(
        fake_pmids or fake_ncts or fake_docs or check.get("invalid_brackets")
    )
    if suspicious:
        try:
            from src.llm import get_llm, with_json_mode_chat

            llm = get_llm()
            if not llm.is_offline:
                rewritten = _llm_rewrite_citations(answer, contexts)
                if rewritten and verify_citations(rewritten, contexts).get("ok"):
                    return rewritten
        except Exception:
            pass
    return fixed


def _llm_rewrite_citations(
    answer: str,
    contexts: list[dict[str, Any]],
) -> str | None:
    """
    基于合法证据，让 LLM 重写回答中的引用部分（结构化 JSON 输出）。

    参数:
        answer: 原始回答。
        contexts: 当次检索证据。

    返回:
        str | None: 重写后的回答；失败返回 None。
    """
    from src.llm import with_json_mode_chat

    evidence_lines = []
    for i, c in enumerate(contexts, start=1):
        evidence_lines.append(
            f"[{i}] {c.get('title') or '无标题'} | {c.get('source') or ''} | "
            f"{c.get('year') or 'n/a'} | {c.get('doc_id') or ''}\n"
            f"{(c.get('text') or '')[:240]}"
        )
    payload = with_json_mode_chat(
        [
            {
                "role": "system",
                "content": (
                    "你是引用修复器。回答中所有 [n] 必须来自给定证据列表；"
                    "禁止编造 PMID/NCT/文献 ID。保持原回答内容与语气，只修正引用。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原始回答：\n{answer}\n\n"
                    f"合法证据列表：\n" + "\n\n".join(evidence_lines) + "\n\n"
                    '输出 JSON：{"answer": "重写后的回答", "citations": [用到的合法编号]}'
                ),
            },
        ],
        temperature=0.0,
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str):
        return None
    return payload["answer"].strip() or None


def detect_unsupported_claims(
    answer: str,
    contexts: list[dict[str, Any]],
) -> list[str]:
    """
    找出回答中可能未被证据支撑的句子/主张。

    参数:
        answer: 模型回答。
        contexts: 当次证据。

    返回:
        list[str]: 可疑主张列表（原文句子）。

    作用:
        辅助人工复核与自动拒答/降级策略。
    """
    import re as _re

    # 排除声明/参考文献等结构性文本
    body = _re.split(r"声明|参考文献|---", answer)[0]
    context_text = " ".join(
        str(c.get("text") or "") + " " + str(c.get("title") or "") for c in contexts
    ).lower()
    context_tokens = set(_re.findall(r"[\w\u4e00-\u9fff]+", context_text))

    suspicious: list[str] = []
    for sent in _re.split(r"[。！？!?；;\n]", body):
        sent = sent.strip()
        if len(sent) < 8:
            continue
        # 有合法 [n] 引用的句子默认视为有支撑
        if _re.search(r"\[\d+\]", sent):
            continue
        toks = _re.findall(r"[\w\u4e00-\u9fff]+", sent.lower())
        if not toks:
            continue
        overlap = sum(1 for t in toks if t in context_tokens) / len(toks)
        # 含数字/百分比/年份等「具体主张」且与证据重叠不足
        has_claim_marker = bool(_re.search(r"\d+(\.\d+)?%?|提升|降低|显著|有效|优于", sent))
        if overlap < 0.35 and has_claim_marker:
            suspicious.append(sent)
    return suspicious
