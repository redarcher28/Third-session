# -*- coding: utf-8 -*-
"""
Streamlit 备用演示界面：统一的赛道一/二助手 + 保留赛道三评测入口。

FastAPI 根路径提供同一套前端的轻量 Web 版本；这个文件保留 Streamlit，方便
课堂快速演示和不启动独立前端时直接调用同一条 Python 编排链路。
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
from src.tracks.prompt_profiles import PROMPT_VERSION, get_track_profile


st.set_page_config(
    page_title="证据台 · Evidence Desk",
    page_icon="⌁",
    layout="wide",
)

ETHICS = (
    "⚠️ **伦理声明**：本系统仅供学习与研究演示，**不构成医疗建议**，"
    "不用于真实诊疗，不处理真实患者隐私。引用内容请人工复核原文。"
)


def render_ethics_banner() -> None:
    """统一渲染伦理声明，避免赛道切换时遗漏安全边界。"""
    st.info(ETHICS)


def render_evidence_cards(contexts: list) -> None:
    """统一渲染证据卡片，保留编号、元数据、摘要片段和可回查链接。"""
    if not contexts:
        st.caption("本次没有可展示的证据卡片。")
        return
    for citation in contexts:
        with st.expander(
            f"[{citation.index}] {citation.title} · {citation.evidence_level}",
            expanded=False,
        ):
            st.write(
                f"来源：`{citation.source}` · 年份：{citation.year or '未知'} · "
                f"`{citation.doc_id}`"
            )
            if citation.url:
                st.markdown(f"[打开原始来源]({citation.url})")
            st.write(citation.text or citation.snippet or "暂无摘要片段")


def render_retrieval_explanation(explanation: dict) -> None:
    """展示统一后端返回的检索摘要，而不是暴露内部全部候选。"""
    if not explanation:
        return
    sources = explanation.get("sources") or {}
    levels = explanation.get("evidence_levels") or {}
    with st.expander("为什么是这些证据？", expanded=False):
        st.write(f"检索查询：`{explanation.get('rewritten_query') or '未生成'}`")
        st.write(f"返回证据：{explanation.get('retrieved_count', 0)} 条")
        st.write(f"来源分布：{sources or '暂无'}")
        st.write(f"证据等级：{levels or '暂无'}")
        st.caption(
            "Prompt 栈：query reformulation → grounded system → synthesis → "
            "citation validation"
        )


def render_assistant(track: str, sample_questions: list[str]) -> None:
    """渲染统一问答工作区；赛道差异来自后端预置 Prompt 配置。"""
    profile = get_track_profile(track)
    st.markdown(f"### {profile.label}")
    st.caption(f"面向：{profile.audience} · {profile.description}")

    col_q, col_opt = st.columns([3, 1])
    with col_q:
        default = st.session_state.get(f"q_{track}", sample_questions[0] if sample_questions else "")
        q = st.text_area("输入问题", value=default, height=125, key=f"q_{track}")
    with col_opt:
        live = st.checkbox("启用在线补检索", value=False, key=f"live_{track}")
        top_k = st.slider("证据条数", 3, 8, 5, key=f"k_{track}")
        sample = st.selectbox("示例问题", sample_questions, key=f"sample_{track}")
        if st.button("填入示例", key=f"fill_{track}"):
            st.session_state[f"q_{track}"] = sample
            st.rerun()

    st.caption(f"输出契约：{profile.output_contract} · 禁区：{profile.forbidden_contract}")

    if not st.button("生成带引用回答", type="primary", key=f"go_{track}"):
        return
    if not q.strip():
        st.warning("请先输入问题。")
        return

    with st.spinner("改写查询、检索证据并进行引用校验…"):
        try:
            response = ask(
                q.strip(),
                track=track,
                top_k=top_k,
                use_live_tools=live,
            )
        except Exception as exc:
            st.error(f"调用失败：{exc}")
            return

    st.divider()
    st.subheader("回答")
    if response.refused:
        st.warning(response.answer)
    else:
        st.markdown(response.answer)

    check = response.citation_check or {}
    status = "✅ 通过" if check.get("ok") else "⚠️ 存疑"
    st.write(f"引用校验：{status} · Prompt：`{response.prompt_version or PROMPT_VERSION}`")
    render_retrieval_explanation(response.retrieval)

    st.subheader("证据面板")
    render_evidence_cards(response.contexts)


st.title("证据台 · Evidence Desk")
st.caption("OpenEvidence 风格 · 同一检索纪律，适配临床证据与健康营养两种语言层")
render_ethics_banner()

tab_assistant, tab_eval = st.tabs(["统一助手 · 赛道一 / 二", "赛道三 · RAG vs Baseline"])

with tab_assistant:
    track_label = st.radio(
        "选择回答视角",
        options=["clinical", "nutrition"],
        format_func=lambda key: get_track_profile(key).label,
        horizontal=True,
        key="unified_track",
    )
    profile = get_track_profile(track_label)
    st.markdown(
        f"**语言层：** {profile.language_contract}　·　"
        f"**输出：** {profile.output_contract}"
    )
    render_assistant(track_label, list(profile.sample_questions))

with tab_eval:
    st.markdown(
        "对比 **纯通用大模型（Baseline）** 与 **带知识库的 RAG**，"
        "用于观察引用、覆盖率和拒答边界。"
    )
    results_path = ROOT / "data" / "eval" / "results" / "benchmark_results.json"
    if st.button("运行评测（可能较慢）", type="primary"):
        with st.spinner("正在跑 Baseline 与 RAG…"):
            payload = run_benchmark()
        st.success("评测完成")
    elif results_path.exists():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        st.caption("当前展示已落盘的最近一次评测结果。")
    else:
        payload = None
        st.info("尚无评测结果，点击上方按钮运行。")

    if payload:
        summary = payload["summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RAG 假引用率", summary["rag_fake_citation_rate"])
        m2.metric("RAG 引用覆盖率", summary["rag_citation_coverage"])
        m3.metric("RAG 拒答率", summary["rag_refusal_rate"])
        m4.metric("RAG 要点覆盖", summary["rag_avg_gold_coverage"])

        chart_df = pd.DataFrame(
            {
                "指标": ["假引用相关", "要点覆盖"],
                "RAG": [summary["rag_fake_citation_rate"], summary["rag_avg_gold_coverage"]],
                "Baseline": [
                    summary["baseline_fake_citation_signal_rate"],
                    summary["baseline_avg_gold_coverage"],
                ],
            }
        ).set_index("指标")
        st.bar_chart(chart_df)

        rows = [
            {
                "id": item.get("id"),
                "track": item["track"],
                "RAG 假引用": item["rag"]["fake_citation_count"],
                "Baseline 信号": item["baseline"]["fake_citation_signal"],
                "RAG 覆盖": item["rag"]["gold_coverage"],
                "Baseline 覆盖": item["baseline"]["gold_coverage"],
                "RAG 拒答": item["rag"]["refused"],
            }
            for item in payload["results"]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("典型 Case")
        for item in payload["results"]:
            with st.expander(f"{item.get('id')} · {item['question'][:46]}…"):
                st.markdown("**RAG**")
                st.write(item["rag"]["answer"][:1200])
                st.markdown("**Baseline**")
                st.write(item["baseline"]["answer"][:1200])

st.divider()
st.caption(f"{ETHICS} · Prompt 栈版本：{PROMPT_VERSION}")
