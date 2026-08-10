# -*- coding: utf-8 -*-
"""
Streamlit 演示界面：三 Tab（临床 / 营养 / 评测）。

运行：
    streamlit run src/app/ui.py
"""

from __future__ import annotations

import json
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

st.title("OpenEvidence 风格 · 证据智能助手 MVP")
st.caption("三赛道：临床证据助手 · 健康营养助手 · RAG vs 通用大模型评测")
st.info(ETHICS)

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
    col_q, col_opt = st.columns([3, 1])
    with col_q:
        default = sample_questions[0] if sample_questions else ""
        q = st.text_area("输入问题", value=default, height=100, key=f"q_{track}")
    with col_opt:
        live = st.checkbox("启用在线补检索", value=False, key=f"live_{track}")
        top_k = st.slider("证据条数", 3, 8, 5, key=f"k_{track}")
        st.selectbox("示例问题", sample_questions, key=f"sample_{track}")
        if st.session_state.get(f"sample_{track}"):
            # sync button
            if st.button("填入示例", key=f"fill_{track}"):
                st.session_state[f"q_{track}"] = st.session_state[f"sample_{track}"]
                st.rerun()

    if st.button("生成带引用回答", type="primary", key=f"go_{track}"):
        with st.spinner("检索证据并生成中…"):
            try:
                resp = ask(
                    st.session_state.get(f"q_{track}", q),
                    track=track,
                    top_k=top_k,
                    use_live_tools=live,
                )
            except Exception as e:
                st.error(f"调用失败: {e}")
                st.stop()
        st.subheader("回答")
        if resp.refused:
            st.warning(resp.answer)
        else:
            st.markdown(resp.answer)
        st.caption(f"改写查询：{resp.rewritten_query}")
        check = resp.citation_check or {}
        if check:
            st.write(
                "引用校验：",
                "✅ 通过" if check.get("ok") else "⚠️ 存疑",
                check,
            )

        st.subheader("证据面板")
        for c in resp.contexts:
            with st.expander(f"[{c.index}] {c.title} ({c.evidence_level})"):
                st.write(f"来源：`{c.source}` · 年份：{c.year or 'n/a'} · `{c.doc_id}`")
                if c.url:
                    st.write(c.url)
                st.write(c.snippet)


with tab_clinical:
    st.markdown(
        "面向医生 / 医学生：结构化证据回答，优先指南 / 荟萃 / RCT。"
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


# ---------------------------------------------------------------------------
# 【待完善】UI 组件拆分（只定义签名与备注，不写函数体）
# ---------------------------------------------------------------------------


def render_ethics_banner() -> None:
    """
    【待完善】Streamlit 统一伦理声明横幅组件。

    参数:
        无

    返回:
        None（副作用：在页面渲染声明）。

    作用:
        保证各 Tab 伦理提示一致，避免演示遗漏。
    """
    raise NotImplementedError("待队员实现：render_ethics_banner")


def render_evidence_cards(contexts: list) -> None:
    """
    【待完善】独立渲染证据卡片列表。

    参数:
        contexts: Citation 列表或等价结构。

    返回:
        None（副作用：在页面展示卡片）。

    作用:
        把 UI 展示与问答逻辑解耦，方便多人并行改界面。
    """
    raise NotImplementedError("待队员实现：render_evidence_cards")


def render_retrieval_explanation(explanation: dict) -> None:
    """
    【待完善】展示检索可解释信息面板。

    参数:
        explanation: explain_retrieval 返回的字典。

    返回:
        None（副作用：页面渲染）。

    作用:
        让演示观众看到「为何命中这些证据」。
    """
    raise NotImplementedError("待队员实现：render_retrieval_explanation")
