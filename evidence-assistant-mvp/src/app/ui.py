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

from src.config import validate_runtime_config
from src.tracks.clinical import build_clinical_answer_outline
from src.tracks.eval_bench import (
    compare_metric_delta,
    export_human_rubric_template,
    pick_typical_cases,
    run_benchmark,
)
from src.tracks.nutrition import build_nutrition_action_tips
from src.tracks.pipeline import ask, detect_track_from_question

st.set_page_config(
    page_title="证据智能助手 MVP",
    page_icon="📚",
    layout="wide",
)

ETHICS = (
    "⚠️ **伦理声明**：本系统仅供学习与研究演示，**不构成医疗建议**，"
    "不用于真实诊疗，不处理真实患者隐私。引用内容请人工复核原文。"
)


def render_ethics_banner() -> None:
    """Streamlit 统一伦理声明横幅组件。"""
    st.info(ETHICS)


def _field(obj, name: str, default=""):
    """从 Citation 对象或 dict 中读取字段。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _prepare_markdown(md: str) -> str:
    """清洗模型输出，保证 Streamlit Markdown 能正常渲染。

    处理：
      - 字面量 ``\\n`` 转真实换行
      - 去掉包住全文的 ``` / ```markdown 围栏（声明追加后围栏常不在首尾）
      - 转义 ``$``，避免被当成 LaTeX
      - ``#`` 标题降级为加粗，避免超大字号
      - 列表/标题前补空行，便于 GFM 解析
    """
    if not md:
        return ""
    text = md.replace("\r\n", "\n").replace("\r", "\n")
    # 部分接口会把换行逃逸成字面量 \n
    if "\\n" in text and text.count("\n") <= max(1, text.count("\\n") // 4):
        text = text.replace("\\n", "\n")

    # 去掉包住正文的 markdown/code 围栏（可出现在文中，且声明可能在围栏外）
    text = re.sub(
        r"^\s*```(?:markdown|md|text|plain)?\s*\n",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\n```(?:\s*\n|\s*$)", "\n", text, count=1)
    # 残留的成对围栏：整段删掉开闭标记，保留内部
    text = re.sub(
        r"```(?:markdown|md|text|plain)?\s*\n([\s\S]*?)\n```",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    # 孤立围栏行直接丢掉，避免后续全文落入代码块
    text = re.sub(r"(?m)^\s*```.*$", "", text)

    # Streamlit 默认把 $...$ 当 LaTeX；医学文本里的 $ 需转义
    text = text.replace("$", r"\$")

    out: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if not m:
            out.append(line)
            continue
        rest = m.group(2).strip()
        # 中文标题常用全角空格或冒号，优先整行加粗
        if len(rest) <= 48 or " " not in rest:
            out.append("**" + rest + "**")
            continue
        head, _, body = rest.partition(" ")
        if len(head) > 24 or not body:
            out.append("**" + rest + "**")
        else:
            out.append("**" + head + "**")
            out.append(body)

    # 列表与分隔线前补空行，避免挤成一整段
    fixed: list[str] = []
    for line in out:
        s = line.lstrip()
        needs_blank = bool(re.match(r"^([-*+] |\d+\. |---+\s*$)", s))
        if needs_blank and fixed and fixed[-1].strip():
            fixed.append("")
        fixed.append(line)
    # 压缩过多空行
    cleaned: list[str] = []
    blank = 0
    for line in fixed:
        if not line.strip():
            blank += 1
            if blank <= 2:
                cleaned.append("")
            continue
        blank = 0
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _render_answer(answer: str, *, refused: bool = False) -> None:
    """用 Markdown 渲染回答；拒答时用警告样式容器包裹。"""
    body = _prepare_markdown(answer)
    if refused:
        st.warning("已拒答或证据不足")
        st.markdown(body)
    else:
        st.markdown(body)


def render_evidence_cards(contexts: list) -> None:
    """独立渲染证据卡片列表。"""
    for c in contexts:
        index = _field(c, "index", "")
        title = str(_field(c, "title", "未命名证据"))
        level = _field(c, "evidence_level", "other")
        source = _field(c, "source", "")
        year = _field(c, "year", None)
        doc_id = _field(c, "doc_id", "")
        url = _field(c, "url", "")
        snippet = _field(c, "snippet", _field(c, "text", ""))
        prefix = "Wiki 总览 · " if _field(c, "kind") == "wiki" else ""
        label = (
            f"{prefix}[{index}] {title} ({level})"
            if index != ""
            else f"{prefix}{title} ({level})"
        )
        with st.expander(label):
            st.write(f"来源：`{source}` · 年份：{year or 'n/a'} · `{doc_id}`")
            if url:
                st.write(url)
            st.write(snippet)


def render_retrieval_explanation(explanation: dict) -> None:
    """展示检索可解释信息面板。"""
    if not explanation:
        return
    st.subheader("检索解释")
    for item in explanation.get("why_selected") or []:
        st.markdown(f"- {item}")
    sources = explanation.get("sources") or []
    if sources:
        st.markdown("来源：" + "、".join(f"`{s}`" for s in sources))
    if explanation.get("notes"):
        st.caption(explanation["notes"])


def _render_response_details(msg: dict, default_track: str) -> None:
    """渲染单条助手消息附带的引用校验、证据面板与赛道小结。"""
    resp = msg.get("resp")
    if resp is None:
        return
    track = msg.get("track") or default_track
    st.caption(f"改写查询：{resp.rewritten_query}")
    check = resp.citation_check or {}
    if check:
        with st.expander("引用校验与检索解释"):
            st.write("引用校验：", "✅ 通过" if check.get("ok") else "⚠️ 存疑")
            notes = []
            if check.get("invalid_brackets"):
                notes.append(f"无效编号 {check['invalid_brackets']}")
            if check.get("fake_pmids"):
                notes.append(f"无法核实 PMID {check['fake_pmids']}")
            if check.get("fake_ncts"):
                notes.append(f"无法核实 NCT {check['fake_ncts']}")
            if check.get("fake_docs"):
                notes.append(f"无法核实文献 {check['fake_docs']}")
            if check.get("citation_density_ok") is False:
                notes.append("引用密度不足")
            if notes:
                st.caption("；".join(notes))
            unsupported = check.get("unsupported_claims") or []
            if unsupported:
                st.caption(f"未引用句 {len(unsupported)} 条，请人工复核")
            render_retrieval_explanation(check.get("explanation"))
    if not resp.contexts:
        return
    ctx_dicts = [
        c.model_dump() if hasattr(c, "model_dump") else dict(c)
        for c in resp.contexts
    ]
    with st.expander("证据面板"):
        render_evidence_cards(resp.contexts)
    if track == "clinical":
        outline = msg.get("outline")
        if outline is None:
            outline = build_clinical_answer_outline(ctx_dicts)
            msg["outline"] = outline
        with st.expander("临床大纲"):
            st.markdown(f"**结论**：{outline['conclusion']}")
            st.markdown("**证据等级**：" + "、".join(outline["evidence_levels"]))
            st.markdown("**关键研究/指南**")
            for s in outline["key_studies"]:
                st.markdown(f"- {s}")
            st.markdown("**局限**")
            for s in outline["limitations"]:
                st.markdown(f"- {s}")
    else:
        tips = msg.get("tips")
        if tips is None:
            tips = build_nutrition_action_tips(ctx_dicts)
            msg["tips"] = tips
        with st.expander("你可以怎么做"):
            for tip in tips:
                st.markdown(f"- {tip}")


st.title("OpenEvidence 风格 · 证据智能助手 MVP")
st.caption("三赛道：临床证据助手 · 健康营养助手 · RAG vs 通用大模型评测")
render_ethics_banner()
_runtime = validate_runtime_config()
if _runtime.get("issues"):
    st.caption("运行时提示：" + "；".join(_runtime["issues"]))

tab_clinical, tab_nutrition, tab_eval = st.tabs(
    ["赛道一 · 临床证据助手", "赛道二 · 健康营养助手", "赛道三 · 对比评测"]
)


def render_assistant(track: str, sample_questions: list[str]) -> None:
    """
    渲染单个赛道 Tab：检索选项 + 聊天式多轮对话。

    参数:
        track: "clinical" 或 "nutrition"。
        sample_questions: 示例问题列表。

    返回:
        None（页面副作用渲染）。
    """
    chat_key = f"chat_{track}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    with st.expander("检索选项", expanded=False):
        c1, c2, c3 = st.columns(3)
        live = c1.checkbox("启用在线补检索", value=False, key=f"live_{track}")
        top_k = c2.slider("证据条数", 3, 8, 5, key=f"k_{track}")
        route_choice = c3.selectbox(
            "赛道路由",
            ["跟随自动识别", "clinical", "nutrition"],
            index=0,
            key=f"route_{track}",
        )
        c4, c5, _ = st.columns(3)
        year_options = ["不限"] + [str(y) for y in range(2026, 1989, -1)]
        year_from_choice = c4.selectbox(
            "证据起始年份", year_options, index=0, key=f"yf_{track}"
        )
        year_to_choice = c5.selectbox(
            "证据截止年份", year_options, index=0, key=f"yt_{track}"
        )
        year_from = None if year_from_choice == "不限" else int(year_from_choice)
        year_to = None if year_to_choice == "不限" else int(year_to_choice)

    messages = st.session_state[chat_key]
    pending = None
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                _render_answer(msg["content"], refused=msg.get("refused", False))
                _render_response_details(msg, track)

    if not messages:
        cols = st.columns(min(3, len(sample_questions)) or 1)
        for i, s in enumerate(sample_questions):
            if cols[i % len(cols)].button(
                s, key=f"sample_{track}_{i}", width="stretch"
            ):
                pending = s

    if messages:
        if st.button("清空对话", key=f"clear_{track}"):
            st.session_state[chat_key] = []
            st.rerun()

    if pending is None:
        pending = st.chat_input("输入你的医学问题…", key=f"input_{track}")

    if pending:
        st.session_state[chat_key].append({"role": "user", "content": pending})
        history = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        with st.spinner("检索证据并生成中…"):
            try:
                effective_track = (
                    detect_track_from_question(pending, history=history)
                    if route_choice == "跟随自动识别"
                    else route_choice
                )
                resp = ask(
                    pending,
                    track=effective_track,
                    top_k=top_k,
                    use_live_tools=live,
                    year_from=year_from,
                    year_to=year_to,
                    history=history,
                )
            except Exception as e:
                st.error(f"调用失败: {e}")
                st.stop()
        item: dict = {
            "role": "assistant",
            "content": resp.answer,
            "refused": resp.refused,
            "resp": resp,
            "track": effective_track,
        }
        ctx_dicts = [
            c.model_dump() if hasattr(c, "model_dump") else dict(c)
            for c in resp.contexts
        ]
        if resp.contexts:
            if effective_track == "clinical":
                item["outline"] = build_clinical_answer_outline(ctx_dicts)
            else:
                item["tips"] = build_nutrition_action_tips(ctx_dicts)
        st.session_state[chat_key].append(item)
        st.rerun()


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
        c3, c4 = st.columns(2)
        c3.metric("RAG 平均忠实度", summary.get("rag_avg_faithfulness", 0.0))
        c4.metric("评测题数", summary["n"])
        delta = compare_metric_delta(summary)
        st.caption(
            f"RAG vs Baseline：假引用差值 {delta['fake_citation_delta']:+.3f}，"
            f"要点覆盖差值 {delta['gold_coverage_delta']:+.3f}，"
            f"拒答率 {delta['refusal_rate']:.3f}，引用覆盖率 {delta['citation_coverage']:.3f}"
        )

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
                    "rag_faith": r["rag"].get("faithfulness", 0.0),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("典型 Case")
        for r in pick_typical_cases(payload["results"], n=3):
            with st.expander(f"{r.get('id')} · {r['question'][:40]}…"):
                st.markdown("**RAG**")
                _render_answer(r["rag"]["answer"][:1200])
                st.markdown("**Baseline**")
                _render_answer(r["baseline"]["answer"][:1200])

        if st.button("导出人工评分量表"):
            out = export_human_rubric_template(
                ROOT / "data" / "eval" / "results" / "human_rubric.md"
            )
            st.success(f"已写入 {out}")

st.divider()
st.caption(ETHICS)


