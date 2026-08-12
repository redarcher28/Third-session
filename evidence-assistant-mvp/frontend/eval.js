const runEvalBtn = document.getElementById("runEvalBtn");
const evalStatus = document.getElementById("evalStatus");
const evalEmpty = document.getElementById("evalEmpty");
const evalDashboard = document.getElementById("evalDashboard");
const evalMetrics = document.getElementById("evalMetrics");
const evalChart = document.getElementById("evalChart");
const evalTableBody = document.getElementById("evalTableBody");
const evalCases = document.getElementById("evalCases");

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = String(s ?? "");
  return d.innerHTML;
}

function fmtRate(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return (Number(v) * 100).toFixed(1) + "%";
}

function fmtNum(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return Number(v).toFixed(3);
}

function renderMetrics(summary) {
  if (!evalMetrics || !summary) return;
  const items = [
    { label: "RAG 假引用率", value: fmtRate(summary.rag_fake_citation_rate) },
    { label: "Baseline 假引用信号率", value: fmtRate(summary.baseline_fake_citation_signal_rate) },
    { label: "RAG 引用覆盖率", value: fmtRate(summary.rag_citation_coverage) },
    { label: "RAG 拒答率", value: fmtRate(summary.rag_refusal_rate) },
    { label: "RAG 要点覆盖", value: fmtNum(summary.rag_avg_gold_coverage) },
    { label: "Baseline 要点覆盖", value: fmtNum(summary.baseline_avg_gold_coverage) },
  ];
  evalMetrics.innerHTML = items
    .map(
      (m) => `
    <div class="eval-metric-card">
      <div class="eval-metric-label">${escapeHtml(m.label)}</div>
      <div class="eval-metric-value">${escapeHtml(m.value)}</div>
    </div>`
    )
    .join("");
}

function renderChart(summary) {
  if (!evalChart || !summary) return;
  const rows = [
    {
      label: "假引用相关",
      rag: Number(summary.rag_fake_citation_rate) || 0,
      baseline: Number(summary.baseline_fake_citation_signal_rate) || 0,
    },
    {
      label: "要点覆盖",
      rag: Number(summary.rag_avg_gold_coverage) || 0,
      baseline: Number(summary.baseline_avg_gold_coverage) || 0,
    },
  ];
  const maxVal = Math.max(0.01, ...rows.flatMap((r) => [r.rag, r.baseline]));
  evalChart.innerHTML = rows
    .map((row) => {
      const ragPct = (row.rag / maxVal) * 100;
      const basePct = (row.baseline / maxVal) * 100;
      return `
      <div class="eval-chart-row">
        <div class="eval-chart-label">${escapeHtml(row.label)}</div>
        <div class="eval-chart-bars">
          <div class="eval-bar-group">
            <span class="eval-bar-tag">RAG</span>
            <div class="eval-bar-track"><div class="eval-bar eval-bar--rag" style="width:${ragPct}%"></div></div>
            <span class="eval-bar-num">${escapeHtml(fmtNum(row.rag))}</span>
          </div>
          <div class="eval-bar-group">
            <span class="eval-bar-tag">Baseline</span>
            <div class="eval-bar-track"><div class="eval-bar eval-bar--baseline" style="width:${basePct}%"></div></div>
            <span class="eval-bar-num">${escapeHtml(fmtNum(row.baseline))}</span>
          </div>
        </div>
      </div>`;
    })
    .join("");
}

function renderTable(results) {
  if (!evalTableBody) return;
  evalTableBody.innerHTML = (results || [])
    .map((r) => {
      const rag = r.rag || {};
      const base = r.baseline || {};
      return `
      <tr>
        <td>${escapeHtml(r.id)}</td>
        <td>${escapeHtml(r.track)}</td>
        <td>${escapeHtml(rag.fake_citation_count)}</td>
        <td>${escapeHtml(base.fake_citation_signal)}</td>
        <td>${escapeHtml(fmtNum(rag.gold_coverage))}</td>
        <td>${escapeHtml(fmtNum(base.gold_coverage))}</td>
        <td>${rag.refused ? "是" : "否"}</td>
        <td>${escapeHtml(rag.n_contexts)}</td>
      </tr>`;
    })
    .join("");
}

function renderCases(results) {
  if (!evalCases) return;
  evalCases.innerHTML = (results || [])
    .map((r) => {
      const q = r.question || "";
      const title = `${r.id || ""} · ${q.slice(0, 40)}${q.length > 40 ? "…" : ""}`;
      const ragAns = (r.rag?.answer || "").slice(0, 1200);
      const baseAns = (r.baseline?.answer || "").slice(0, 1200);
      return `
      <details class="eval-case-expander">
        <summary>${escapeHtml(title)}</summary>
        <div class="eval-case-body">
          <p class="eval-case-heading">RAG</p>
          <pre class="eval-case-text">${escapeHtml(ragAns)}</pre>
          <p class="eval-case-heading">Baseline</p>
          <pre class="eval-case-text">${escapeHtml(baseAns)}</pre>
        </div>
      </details>`;
    })
    .join("");
}

function renderPayload(payload) {
  if (!payload || !payload.summary) {
    if (evalDashboard) evalDashboard.hidden = true;
    if (evalEmpty) {
      evalEmpty.hidden = false;
      evalEmpty.textContent =
        "尚未有评测结果。请先运行 python scripts/build_kb.py --skip-live，再点击运行评测。";
    }
    return;
  }
  if (evalEmpty) evalEmpty.hidden = true;
  if (evalDashboard) evalDashboard.hidden = false;
  renderMetrics(payload.summary);
  renderChart(payload.summary);
  renderTable(payload.results);
  renderCases(payload.results);
}

async function loadCachedResults() {
  try {
    const resp = await fetch("/eval/results");
    if (!resp.ok) {
      if (evalEmpty) {
        evalEmpty.hidden = false;
        evalEmpty.textContent =
          "尚未有评测结果。请先运行 python scripts/build_kb.py --skip-live，再点击运行评测。";
      }
      return;
    }
    const data = await resp.json();
    renderPayload(data);
    if (evalStatus) evalStatus.textContent = "已加载最近一次评测结果。";
  } catch {
    if (evalEmpty) {
      evalEmpty.hidden = false;
      evalEmpty.textContent = "加载缓存结果失败，请尝试重新运行评测。";
    }
  }
}

runEvalBtn?.addEventListener("click", async () => {
  runEvalBtn.disabled = true;
  if (evalStatus) evalStatus.textContent = "正在跑 Baseline 与 RAG…";
  if (evalEmpty) evalEmpty.hidden = true;
  try {
    const resp = await fetch("/eval/run", { method: "POST" });
    const raw = await resp.text();
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      data = null;
    }
    if (!resp.ok) {
      throw new Error(typeof data === "object" && data?.detail ? String(data.detail) : `HTTP ${resp.status}`);
    }
    renderPayload(data);
    if (evalStatus) evalStatus.textContent = "评测完成。";
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (window.showErrorModal) window.showErrorModal(msg);
    if (evalEmpty) {
      evalEmpty.hidden = false;
      evalEmpty.textContent = `失败：${msg}`;
    }
    if (evalStatus) evalStatus.textContent = "";
  } finally {
    runEvalBtn.disabled = false;
  }
});

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", loadCachedResults);
} else {
  loadCachedResults();
}
