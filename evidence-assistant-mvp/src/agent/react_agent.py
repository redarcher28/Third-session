# -*- coding: utf-8 -*-
"""
ReAct 证据智能体：Thought → Action → Observation 循环调用检索/在线工具，最终 finish 带引用回答。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.config import get_settings
from src.generation.answer import DISCLAIMER, generate_answer
from src.kb.chunking import docs_to_chunks
from src.llm import get_llm
from src.models import Citation
from src.retrieval.hybrid import HybridRetriever
from src.tools.cite_check import strip_invalid_claims, verify_citations
from src.tools.live_search import search_clinical_trials, search_pubmed
from src.tracks.clinical import CLINICAL_PERSONA, CLINICAL_STYLE, PREFER_LEVELS
from src.tracks.nutrition import (
    BOOST_TAGS,
    NUTRITION_DOSAGE_REFUSAL,
    NUTRITION_PERSONA,
    NUTRITION_STYLE,
    detect_dosage_request,
    flag_dosage_in_answer,
    simplify_medical_terms,
)
from src.tracks.pipeline import build_retrieval_summary, reformulate_query
from src.tracks.prompt_profiles import PROMPT_VERSION

logger = logging.getLogger(__name__)

MAX_STEPS = 5

REACT_SYSTEM = """你是「OpenEvidence 风格证据智能助手」的 ReAct 推理模块，面向医学/健康证据检索与带引用回答。

可用工具（每次只调用一个）：
1. retrieve_evidence — 从本地知识库混合检索证据
   输入 JSON: {{"query": "检索词", "top_k": 5}}
2. search_pubmed — 在线 PubMed 补检索（较慢）
   输入 JSON: {{"query": "检索词", "retmax": 3}}
3. search_trials — 在线 ClinicalTrials 补检索
   输入 JSON: {{"condition": "Hypertension", "page_size": 2}}
4. finish — 结束推理并给出最终回答（必须基于已 Observation 中的证据，带 [n] 引用）
   输入 JSON: {{"answer": "完整中文回答"}}

严格按以下格式输出（不要省略标签名）：
Thought: （一句话说明下一步为何这样做）
Action: （工具名，四选一）
Action Input: （单行 JSON）

规则：
- 先检索再 finish；证据不足时 finish 中说明「当前知识库证据不足」。
- 禁止编造 PMID/NCT/文献；引用编号必须对应 Observation 中列出的 [n]。
- 本系统不构成医疗建议；finish 的回答末尾需提醒引用需人工复核。
- 赛道人格：{persona_hint}
"""


def _parse_react(text: str) -> tuple[str, str, dict[str, Any]]:
    thought = ""
    action = ""
    action_input: dict[str, Any] = {}
    m_th = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.S | re.I)
    if m_th:
        thought = m_th.group(1).strip()
    m_ac = re.search(r"Action:\s*(\w+)", text, re.I)
    if m_ac:
        action = m_ac.group(1).strip().lower()
    m_in = re.search(r"Action Input:\s*(\{.*\})", text, re.S | re.I)
    if m_in:
        try:
            action_input = json.loads(m_in.group(1))
        except json.JSONDecodeError:
            action_input = {"query": m_in.group(1).strip().strip('"')}
    return thought, action, action_input


def _contexts_to_citation_list(contexts: list[dict[str, Any]]) -> list[Citation]:
    out: list[Citation] = []
    for i, c in enumerate(contexts, start=1):
        out.append(
            Citation(
                index=i,
                doc_id=str(c.get("doc_id") or ""),
                title=str(c.get("title") or ""),
                source=str(c.get("source") or ""),
                year=None if c.get("year") in (None, -1, "-1") else int(c["year"]),
                url=str(c.get("url") or ""),
                evidence_level=str(c.get("evidence_level") or "other"),
                snippet=str(c.get("text") or "")[:240],
            )
        )
    return out


def _format_observation(contexts: list[dict[str, Any]], prefix: str = "") -> str:
    if not contexts:
        return prefix + "未检索到相关证据。"
    lines = [prefix + f"共 {len(contexts)} 条证据："]
    for i, c in enumerate(contexts, start=1):
        lines.append(
            f"[{i}] {c.get('title', '')} ({c.get('evidence_level', '')}) "
            f"| {c.get('source', '')} | {c.get('doc_id', '')}\n"
            f"{str(c.get('text', ''))[:300]}"
        )
    return "\n\n".join(lines)


class ReactEvidenceAgent:
    """ReAct 证据助手：多步工具调用 + 最终带引用回答。"""

    def __init__(
        self,
        track: str = "clinical",
        *,
        top_k: int = 5,
        use_live_tools: bool = False,
    ) -> None:
        self.track = "nutrition" if track == "nutrition" else "clinical"
        self.top_k = max(3, min(8, top_k))
        self.use_live_tools = use_live_tools
        self.retriever = HybridRetriever()
        self.contexts: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []
        self.rewritten_query = ""
        self.query_reformulation_mode = "lexical"

    def _persona(self) -> tuple[str, str]:
        if self.track == "nutrition":
            return NUTRITION_PERSONA, NUTRITION_STYLE
        return CLINICAL_PERSONA, CLINICAL_STYLE

    def _retrieve(self, query: str, top_k: int | None = None) -> str:
        k = top_k if top_k is not None else self.top_k
        prefer = PREFER_LEVELS if self.track == "clinical" else None
        boost = BOOST_TAGS if self.track == "nutrition" else None
        hits = self.retriever.retrieve(
            query,
            top_k=k,
            prefer_levels=prefer,
            boost_tags=boost,
            use_llm_rerank=get_settings().rag_use_llm_rerank,
        )
        seen = {str(c.get("doc_id")) for c in self.contexts}
        for h in hits:
            if str(h.get("doc_id")) not in seen:
                self.contexts.append(h)
                seen.add(str(h.get("doc_id")))
        return _format_observation(hits, "本地知识库检索结果：\n")

    def _pubmed(self, query: str, retmax: int = 3) -> str:
        try:
            docs = search_pubmed(query, retmax=retmax)
            chunks = [docs_to_chunks([d])[0].model_dump() for d in docs]
            seen = {str(c.get("doc_id")) for c in self.contexts}
            for c in chunks:
                if str(c.get("doc_id")) not in seen:
                    self.contexts.append(c)
                    seen.add(str(c.get("doc_id")))
            return _format_observation(chunks, "PubMed 在线检索：\n")
        except Exception as e:
            return f"PubMed 检索失败：{e}"

    def _trials(self, condition: str, page_size: int = 2) -> str:
        try:
            docs = search_clinical_trials(condition, page_size=page_size)
            chunks = [docs_to_chunks([d])[0].model_dump() for d in docs]
            seen = {str(c.get("doc_id")) for c in self.contexts}
            for c in chunks:
                if str(c.get("doc_id")) not in seen:
                    self.contexts.append(c)
                    seen.add(str(c.get("doc_id")))
            return _format_observation(chunks, "ClinicalTrials 在线检索：\n")
        except Exception as e:
            return f"临床试验检索失败：{e}"

    def _execute(self, action: str, action_input: dict[str, Any]) -> str:
        action = action.lower().replace("-", "_")
        if action == "retrieve_evidence":
            q = str(action_input.get("query") or action_input.get("q") or "")
            top_k = int(action_input.get("top_k") or 5)
            return self._retrieve(q, top_k=top_k)
        if action == "search_pubmed":
            q = str(action_input.get("query") or "")
            retmax = int(action_input.get("retmax") or 3)
            return self._pubmed(q, retmax=retmax)
        if action == "search_trials":
            cond = str(action_input.get("condition") or "Hypertension")
            page_size = int(action_input.get("page_size") or 2)
            return self._trials(cond, page_size=page_size)
        if action == "finish":
            return "__FINISH__"
        return f"未知工具：{action}。请使用 retrieve_evidence / search_pubmed / search_trials / finish。"

    def run(self, question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """
        执行 ReAct 循环。

        返回:
            dict: reply, steps, citations, contexts, track, citation_check
        """
        history = history or []
        persona, style = self._persona()
        if self.track == "nutrition" and detect_dosage_request(question):
            return {
                "reply": NUTRITION_DOSAGE_REFUSAL + DISCLAIMER,
                "steps": [],
                "citations": [],
                "contexts": [],
                "track": self.track,
                "rewritten_query": question,
                "refused": True,
                "citation_check": {"ok": True, "has_citations": False, "reason": "dosage_request"},
                "retrieval": build_retrieval_summary(
                    [],
                    rewritten_query=question,
                    top_k=self.top_k,
                    use_live_tools=self.use_live_tools,
                    query_reformulation_mode="guarded",
                ),
                "prompt_version": PROMPT_VERSION,
            }
        self.rewritten_query, self.query_reformulation_mode = reformulate_query(question, self.track)
        rewritten = self.rewritten_query

        llm = get_llm()
        sys_content = REACT_SYSTEM.format(
            persona_hint="营养科普" if self.track == "nutrition" else "临床证据"
        )
        thread: list[dict[str, str]] = [{"role": "system", "content": sys_content}]
        for m in history[-8:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                thread.append({"role": m["role"], "content": m["content"]})
        thread.append(
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n"
                    f"建议检索查询：{rewritten}\n"
                    "请开始 ReAct：先 Thought，再 Action 与 Action Input。"
                ),
            }
        )

        final_answer = ""
        for step in range(MAX_STEPS):
            raw = llm.chat(thread, temperature=0.2, max_tokens=1200)
            thought, action, action_input = _parse_react(raw)
            if not action:
                # 离线或格式异常：先做一次默认检索再兜底生成
                if not self.contexts:
                    self._retrieve(rewritten or question)
                break
            obs = self._execute(action, action_input)
            step_rec = {
                "step": step + 1,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "raw": raw,
            }
            if action == "finish":
                final_answer = str(action_input.get("answer") or "")
                step_rec["observation"] = "（完成）"
                self.steps.append(step_rec)
                break
            step_rec["observation"] = obs[:2000]
            self.steps.append(step_rec)
            thread.append({"role": "assistant", "content": raw})
            thread.append({"role": "user", "content": f"Observation:\n{obs}\n\n请继续 Thought / Action，或 finish。"})

        if self.use_live_tools and rewritten:
            self._pubmed(rewritten, retmax=3)

        refused = False
        citations: list[Citation] = []
        if not final_answer:
            answer, citations, refused = generate_answer(
                question,
                self.contexts,
                system_persona=persona,
                answer_style=style,
            )
            final_answer = answer
        else:
            citations = _contexts_to_citation_list(self.contexts)
            if DISCLAIMER.strip() not in final_answer:
                final_answer = final_answer.rstrip() + DISCLAIMER
            refused = not self.contexts

        check = verify_citations(final_answer, self.contexts)
        final_answer = strip_invalid_claims(final_answer, check)
        if self.track == "nutrition" and not refused:
            final_answer = simplify_medical_terms(final_answer)
            if flag_dosage_in_answer(final_answer):
                final_answer = final_answer.rstrip() + (
                    "\n\n> ⚠️ 如涉及具体药量/剂量，请以医生或药师的意见为准，切勿自行调整。"
                )

        return {
            "reply": final_answer,
            "steps": self.steps,
            "citations": [c.model_dump() for c in citations],
            "contexts": [c.model_dump() for c in _contexts_to_citation_list(self.contexts)],
            "track": self.track,
            "rewritten_query": rewritten,
            "refused": refused,
            "citation_check": check,
            "retrieval": build_retrieval_summary(
                self.contexts,
                rewritten_query=rewritten,
                top_k=self.top_k,
                use_live_tools=self.use_live_tools,
                query_reformulation_mode=self.query_reformulation_mode,
            ),
            "prompt_version": PROMPT_VERSION,
        }


def react_chat(
    messages: list[dict[str, str]],
    *,
    track: str = "clinical",
    top_k: int = 5,
    use_live_tools: bool = False,
) -> dict[str, Any]:
    """从多轮 messages 取最后一条 user 问题并运行 ReAct。"""
    user_msgs = [m for m in messages if m.get("role") == "user" and (m.get("content") or "").strip()]
    if not user_msgs:
        return {
            "reply": "请先输入您的医学/健康问题。",
            "steps": [],
            "contexts": [],
            "citations": [],
            "prompt_version": PROMPT_VERSION,
        }
    question = user_msgs[-1]["content"].strip()
    history = messages[:-1] if len(messages) > 1 else []
    agent = ReactEvidenceAgent(track=track, top_k=top_k, use_live_tools=use_live_tools)
    return agent.run(question, history)
