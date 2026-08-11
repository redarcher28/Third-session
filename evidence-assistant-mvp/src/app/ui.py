# -*- coding: utf-8 -*-
"""
Streamlit 演示界面：三 Tab（临床 / 营养 / 评测）。

运行：
    streamlit run src/app/ui.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tracks.eval_bench import run_benchmark
from src.tracks.pipeline import ask

st.set_page_config(
    page_title="证据智能助手 MVP",
    page_icon="📚",
    layout="wide",
)

ETHICS = (
    "⚠️ **伦理声明**：本系统仅供学习与研究演示，**不构成医疗建议**，"
    "不用于真实诊疗，不处理真实患者隐私。引用内容请人工复核原文。"
)

LEVEL_ICON = {
    "guideline": "🏛️",
    "meta": "📊",
    "rct": "🔬",
    "observational": "📋",
    "wiki": "📖",
    "ebook": "📘",
    "other": "❓",
}


def _get(obj, key, default=None):
    """兼容 dict 与 pydantic 对象的字段读取。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def render_ethics_banner() -> None:
    """
    Streamlit 统一伦理声明横幅组件。

    参数:
        无

    返回:
        None（副作用：在页面渲染声明）。

    作用:
        保证各 Tab 伦理提示一致，避免演示遗漏。
    """
    st.info(ETHICS)


def render_evidence_cards(contexts: list) -> None:
    """
    独立渲染证据卡片列表。

    参数:
        contexts: Citation 列表或等价结构。

    返回:
        None（副作用：在页面展示卡片）。

    作用:
        把 UI 展示与问答逻辑解耦，方便多人并行改界面。
    """
    for c in contexts:
        idx = _get(c, "index")
        title = _get(c, "title") or "未命名证据"
        level = _get(c, "evidence_level") or "other"
        icon = LEVEL_ICON.get(str(level).lower(), "❓")
        source = _get(c, "source") or ""
        year = _get(c, "year")
        doc_id = _get(c, "doc_id") or ""
        url = _get(c, "url") or ""
        snippet = _get(c, "snippet") or _get(c, "text") or ""
        with st.expander(f"{icon} [{idx}] {title} ({level})"):
            st.write(f"来源：`{source}` · 年份：{year or 'n/a'} · `{doc_id}`")
            if url:
                st.write(url)
            st.write(str(snippet)[:240])


def render_anchored_evidence_cards(contexts: list) -> None:
    """
    证据卡片（常驻可见 + 锚点）：供回答内 [n] 点击定位，临床/营养通用。

    参数:
        contexts: Citation 列表或等价结构。

    返回:
        None（副作用：页面渲染）。
    """
    for c in contexts:
        idx = _get(c, "index")
        title = _get(c, "title") or "未命名证据"
        level = _get(c, "evidence_level") or "other"
        icon = LEVEL_ICON.get(str(level).lower(), "❓")
        source = _get(c, "source") or ""
        year = _get(c, "year")
        doc_id = _get(c, "doc_id") or ""
        url = _get(c, "url") or ""
        snippet = _get(c, "snippet") or _get(c, "text") or ""
        st.markdown(f'<a id="cite-{idx}"></a>', unsafe_allow_html=True)
        st.markdown(
            f"{icon} **[{idx}] {title}**　`{source}` · {year or 'n/a'} · "
            f"证据等级 `{level}`"
        )
        st.caption(f"`{doc_id}`" + (f" · 🔗 {url}" if url else ""))
        st.write(str(snippet)[:240])
        st.markdown("---")


def render_retrieval_explanation(explanation: dict) -> None:
    """
    展示检索可解释信息面板。

    参数:
        explanation: explain_retrieval 返回的字典。

    返回:
        None（副作用：页面渲染）。

    作用:
        让演示观众看到「为何命中这些证据」。
    """
    if not explanation:
        return
    with st.expander("🔍 检索过程说明", expanded=False):
        st.write(f"检索查询：`{explanation.get('query') or ''}`")
        why = explanation.get("why_selected") or []
        if why:
            st.markdown("**为什么选中这些证据**")
            for line in why:
                st.markdown(f"- {line}")
        sources = explanation.get("sources") or []
        if sources:
            st.write(f"来源分布：{', '.join(sources)}")
        if explanation.get("notes"):
            st.caption(explanation["notes"])


st.title("OpenEvidence 风格 · 证据智能助手 MVP")
st.caption("三赛道：临床证据助手 · 健康营养助手 · RAG vs 通用大模型评测")
render_ethics_banner()

tab_clinical, tab_nutrition, tab_eval = st.tabs(
    ["赛道一 · 临床证据助手", "赛道二 · 健康营养助手", "赛道三 · 对比评测"]
)


def render_assistant(track: str, sample_questions: list[str]) -> None:
    """
    渲染单个助手 Tab（提问区 + 回答 + 证据面板）。

    参数:
        track: "clinical" 或 "nutrition"。
        sample_questions: 示例问题列表。

    返回:
        None（页面副作用渲染）。
    """
    # 「填入示例」：Streamlit 不允许在 text_area 实例化后修改其 session_state，
    # 因此先处理标志位，在创建文本框之前更新问题内容。
    fill_flag = f"fill_pending_{track}"
    if st.session_state.get(fill_flag):
        sample = st.session_state.get(f"sample_{track}")
        if sample:
            st.session_state[f"q_{track}"] = sample
        st.session_state[fill_flag] = False
        st.rerun()

    col_q, col_opt = st.columns([3, 1])
    with col_q:
        default = sample_questions[0] if sample_questions else ""
        q = st.text_area("输入问题", value=default, height=100, key=f"q_{track}")
    with col_opt:
        live = st.checkbox("启用在线补检索", value=False, key=f"live_{track}")
        top_k = st.slider("证据条数", 3, 8, 5, key=f"k_{track}")
        year_from = None
        high_quality_only = False
        recent = st.checkbox("近5年（2021+）", value=False, key=f"recent_{track}")
        high_quality_only = st.checkbox(
            "仅高质量证据（指南/荟萃/RCT）",
            value=False,
            key=f"hq_{track}",
        )
        if recent:
            year_from = 2021
        st.selectbox("示例问题", sample_questions, key=f"sample_{track}")
        if st.session_state.get(f"sample_{track}"):
            # sync button
            if st.button("填入示例", key=f"fill_{track}"):
                st.session_state[fill_flag] = True
                st.rerun()

    if st.button("生成带引用回答", type="primary", key=f"go_{track}"):
        with st.spinner("检索证据并生成中…"):
            try:
                resp = ask(
                    st.session_state.get(f"q_{track}", q),
                    track=track,
                    top_k=top_k,
                    use_live_tools=live,
                    year_from=year_from,
                    high_quality_only=high_quality_only,
                )
            except Exception as e:
                st.error(f"调用失败: {e}")
                st.stop()
        st.subheader("回答")
        if resp.refused:
            st.warning(resp.answer)
            reason = (resp.citation_check or {}).get("refusal_reason")
            if reason:
                st.caption(f"拒答原因：{reason}")
            suggestions = (resp.citation_check or {}).get("suggestions") or []
            if suggestions:
                st.markdown("**可以这样改进问法**")
                for s in suggestions:
                    st.markdown(f"- {s}")
        else:
            answer_display = resp.answer
            # 把 [n] 变成可点击锚点，点击后定位到对应证据卡片
            answer_display = re.sub(r"\[(\d+)\]", r"[[\1]](#cite-\1)", resp.answer)
            st.markdown(answer_display)
            outline = (resp.citation_check or {}).get(
                "clinical_outline" if track == "clinical" else "nutrition_outline"
            )
            if outline:
                if track == "clinical":
                    st.subheader("临床大纲")
                    st.markdown(f"**结论**：{outline.get('conclusion', '')}")
                    levels = outline.get("evidence_levels") or []
                    if levels:
                        st.write("证据等级：", "、".join(levels))
                    key_studies = outline.get("key_studies") or []
                    if key_studies:
                        st.markdown("**关键研究/指南**")
                        for s in key_studies:
                            st.markdown(f"- {s}")
                    limitations = outline.get("limitations") or []
                    if limitations:
                        st.markdown("**局限**")
                        for lim in limitations:
                            st.markdown(f"- {lim}")
                else:
                    st.subheader("科普结构")
                    st.markdown(f"**通俗结论**：{outline.get('conclusion', '')}")
                    st.markdown(f"**证据一句话**：{outline.get('evidence_sentence', '')}")
                    action_tips = outline.get("action_tips") or []
                    if action_tips:
                        st.markdown("**你可以怎么做**")
                        for tip in action_tips:
                            st.markdown(f"- {tip}")
                    st.markdown(f"**何时就医**：{outline.get('when_to_see_doctor', '')}")
        st.caption(f"改写查询：{resp.rewritten_query}")
        explanation = (resp.citation_check or {}).get("retrieval_explanation")
        if explanation:
            render_retrieval_explanation(explanation)
        check = resp.citation_check or {}
        if check:
            parts = []
            if check.get("invalid_brackets"):
                parts.append(f"无效编号 {check['invalid_brackets']}")
            if check.get("fake_pmids"):
                parts.append(f"无法核实的 PMID {check['fake_pmids']}")
            if check.get("fake_ncts"):
                parts.append(f"无法核实的 NCT {check['fake_ncts']}")
            if check.get("fake_docs"):
                parts.append(f"无法核实的文献 {check['fake_docs']}")
            if check.get("ok"):
                st.caption("✅ 引用校验通过：引用编号与证据面板一致")
            else:
                st.caption("⚠️ 引用存疑：" + ("；".join(parts) if parts else "请人工复核"))

        st.subheader("证据面板")
        render_anchored_evidence_cards(resp.contexts)


with tab_clinical:
    st.markdown(
        "面向医生 / 医学生：结构化证据回答，优先指南 / 荟萃 / RCT。"
    )
    st.caption(
        "输出结构：结论 → 证据等级 → 关键研究/指南 → 局限；"
        "可开启「近5年」与「仅高质量证据」筛选。"
    )
    samples_c = [
        "高血压患者为什么有时要长期吃药？有哪些指南或研究依据？",
        "体检发现血脂偏高，生活方式干预和药物治疗分别有哪些证据？",
        "DASH饮食模式对血压的临床试验证据是什么？",
    ]
    render_assistant("clinical", samples_c)

with tab_nutrition:
    st.markdown(
        "面向普通消费者：通俗科普 + 可追溯引用，强调非诊疗。"
    )
    st.caption(
        "输出结构：通俗结论 → 证据一句话 → 你可以怎么做 → 何时就医；"
        "可开启「近5年」与「仅高质量证据」筛选。"
    )
    samples_n = [
        "地中海饮食对心血管风险有什么证据？",
        "限钠饮食对高血压是否真的有帮助？",
        "血脂高的人日常吃什么更有证据支持？",
    ]
    render_assistant("nutrition", samples_n)

with tab_eval:
    st.markdown(
        "对比 **纯通用大模型（Baseline）** 与 **带知识库的 RAG** 在同一批医学/健康问题上的表现。"
    )
    results_path = ROOT / "data" / "eval" / "results" / "benchmark_results.json"
    if st.button("运行评测（可能较慢）", type="primary"):
        with st.spinner("正在跑 Baseline 与 RAG…"):
            payload = run_benchmark()
        st.success("评测完成")
    elif results_path.exists():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        payload = None
        st.info("尚未有评测结果。请先 `python scripts/build_kb.py --skip-live`，再点击运行评测。")

    if payload:
        summary = payload["summary"]
        st.subheader("汇总指标")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RAG 假引用率", summary["rag_fake_citation_rate"])
        m2.metric("Baseline 假引用信号率", summary["baseline_fake_citation_signal_rate"])
        m3.metric("RAG 引用覆盖率", summary["rag_citation_coverage"])
        m4.metric("RAG 拒答率", summary["rag_refusal_rate"])
        c1, c2 = st.columns(2)
        c1.metric("RAG 要点覆盖", summary["rag_avg_gold_coverage"])
        c2.metric("Baseline 要点覆盖", summary["baseline_avg_gold_coverage"])

        chart_df = pd.DataFrame(
            {
                "指标": ["假引用相关", "要点覆盖"],
                "RAG": [
                    summary["rag_fake_citation_rate"],
                    summary["rag_avg_gold_coverage"],
                ],
                "Baseline": [
                    summary["baseline_fake_citation_signal_rate"],
                    summary["baseline_avg_gold_coverage"],
                ],
            }
        ).set_index("指标")
        st.bar_chart(chart_df)

        st.subheader("逐题结果")
        rows = []
        for r in payload["results"]:
            rows.append(
                {
                    "id": r.get("id"),
                    "track": r["track"],
                    "rag_fake": r["rag"]["fake_citation_count"],
                    "baseline_signal": r["baseline"]["fake_citation_signal"],
                    "rag_cov": r["rag"]["gold_coverage"],
                    "baseline_cov": r["baseline"]["gold_coverage"],
                    "refused": r["rag"]["refused"],
                    "n_contexts": r["rag"]["n_contexts"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("典型 Case")
        for r in payload["results"]:
            with st.expander(f"{r.get('id')} · {r['question'][:40]}…"):
                st.markdown("**RAG**")
                st.write(r["rag"]["answer"][:1200])
                st.markdown("**Baseline**")
                st.write(r["baseline"]["answer"][:1200])

st.divider()
st.caption(ETHICS)
