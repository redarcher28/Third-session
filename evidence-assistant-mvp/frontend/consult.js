const chatLog = document.getElementById("chatLog");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const reactStepsEl = document.getElementById("reactSteps");
const reactEmptyEl = document.getElementById("reactEmpty");
const evidenceListEl = document.getElementById("evidenceList");
const evidenceEmptyEl = document.getElementById("evidenceEmpty");
const evidenceCountEl = document.getElementById("evidenceCount");
const answerMetaPanel = document.getElementById("answerMetaPanel");
const rewrittenQueryLine = document.getElementById("rewrittenQueryLine");
const citationCheckLine = document.getElementById("citationCheckLine");
const promptVersionLine = document.getElementById("promptVersionLine");
const retrievalExpander = document.getElementById("retrievalExpander");
const retrievalExpanderBody = document.getElementById("retrievalExpanderBody");
const pageTitle = document.getElementById("pageTitle");
const pageDesc = document.getElementById("pageDesc");
const trackClinicalBtn = document.getElementById("trackClinical");
const trackNutritionBtn = document.getElementById("trackNutrition");
const useLiveToolsEl = document.getElementById("useLiveTools");
const topKSlider = document.getElementById("topKSlider");
const topKValue = document.getElementById("topKValue");
const sampleSelect = document.getElementById("sampleSelect");
const fillSampleBtn = document.getElementById("fillSampleBtn");
const useReactEl = document.getElementById("useReact");

const ASSISTANT_AVATAR_SRC = "/brand/logo.svg";

/** 与 Streamlit ui.py 示例问题一致 */
const SAMPLE_QUESTIONS = {
  clinical: [
    "高血压患者为什么有时要长期吃药？有哪些指南或研究依据？",
    "体检发现血脂偏高，生活方式干预和药物治疗分别有哪些证据？",
    "DASH饮食模式对血压的临床试验证据是什么？",
  ],
  nutrition: [
    "地中海饮食对心血管风险有什么证据？",
    "限钠饮食对高血压是否真的有帮助？",
    "血脂高的人日常吃什么更有证据支持？",
  ],
};

const TRACK_META = {
  clinical: {
    title: "临床证据咨询",
    desc: "面向医生 / 医学生：结构化证据回答，优先指南 / 荟萃 / RCT。",
    assistantName: "临床证据助手",
    placeholder: "例如：高血压一线降压方案的证据等级如何？",
    welcome: `你好，我是 **临床证据助手**。你可以勾选「启用在线补检索」、调整证据条数，或从示例问题填入后发送。

本页与 B 组 Streamlit 助手能力对齐：带引用回答、改写查询展示、引用校验与可展开证据面板；默认启用 ReAct 多步推理。

**注意**：本回答仅供学习，不构成诊疗建议；引用请人工复核。`,
  },
  nutrition: {
    title: "营养健康咨询",
    desc: "面向普通消费者：通俗科普 + 可追溯引用，强调非诊疗。",
    assistantName: "营养健康助手",
    placeholder: "例如：地中海饮食对心血管有什么好处？",
    welcome: `你好，我是 **营养健康助手**，回答会更通俗，适合健康科普场景。

可启用在线 PubMed 补检索、调整返回证据条数，并从右侧证据面板查看原文片段。

**注意**：不替代医生或营养师个体化建议；引用请人工复核。`,
  },
};

function getTrackFromUrl() {
  const p = new URLSearchParams(window.location.search);
  const t = p.get("track");
  return t === "nutrition" ? "nutrition" : "clinical";
}

let currentTrack = getTrackFromUrl();

/** @type {{ role: 'user'|'assistant', content: string }[]} */
const transcript = [];

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function populateSampleSelect(track) {
  if (!sampleSelect) return;
  const samples = SAMPLE_QUESTIONS[track] || [];
  sampleSelect.innerHTML = samples
    .map((q, i) => `<option value="${i}">${escapeHtml(q.slice(0, 36))}${q.length > 36 ? "…" : ""}</option>`)
    .join("");
}

function applyTrackUI(track) {
  currentTrack = track;
  const meta = TRACK_META[track];
  if (pageTitle) pageTitle.textContent = meta.title;
  if (pageDesc) pageDesc.textContent = meta.desc;
  if (chatInput) chatInput.placeholder = meta.placeholder;
  trackClinicalBtn?.classList.toggle("is-active", track === "clinical");
  trackNutritionBtn?.classList.toggle("is-active", track === "nutrition");
  populateSampleSelect(track);
  const url = new URL(window.location.href);
  url.searchParams.set("track", track);
  window.history.replaceState({}, "", url);
}

function appendBubble(role, content, { loading, refused } = {}) {
  if (!chatLog) return null;
  const wrap = document.createElement("div");
  wrap.className = `consult-msg consult-msg--${role}`;
  const row = document.createElement("div");
  row.className = "consult-msg-row";

  if (role === "assistant") {
    const avatar = document.createElement("img");
    avatar.className = "consult-avatar";
    avatar.src = ASSISTANT_AVATAR_SRC;
    avatar.alt = TRACK_META[currentTrack].assistantName;
    row.appendChild(avatar);
  }

  const body = document.createElement("div");
  body.className = "consult-msg-body";
  const meta = document.createElement("div");
  meta.className = "consult-meta";
  meta.textContent = role === "user" ? "我" : TRACK_META[currentTrack].assistantName;
  const bubble = document.createElement("div");
  bubble.className =
    "consult-bubble" +
    (loading ? " consult-loading" : "") +
    (refused ? " consult-bubble--refused" : "");
  bubble.innerHTML = loading ? "检索证据并生成中…" : escapeHtml(content);
  body.appendChild(meta);
  body.appendChild(bubble);
  row.appendChild(body);
  wrap.appendChild(row);
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
  return wrap;
}

function renderInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function renderAnswerHtml(reply, contexts, citationCheck) {
  const indices = (contexts || []).map((c, i) => Number(c.index ?? i + 1));
  const used = citationCheck?.used_brackets?.length
    ? citationCheck.used_brackets.map(Number)
    : indices;

  const renderLine = (line) =>
    line
      .replace(/\[(\d+)\]/g, (full, num) => {
        const n = Number(num);
        if (!used.includes(n)) return escapeHtml(full);
        return `<a class="cite-ref" href="#evidence-${n}" data-cite="${n}">${escapeHtml(full)}</a>`;
      })
      .split(/(<a class="cite-ref"[^>]*>\[[0-9]+\]<\/a>)/)
      .map((part) => (part.startsWith('<a class="cite-ref"') ? part : renderInlineMarkdown(part)))
      .join("");

  const lines = String(reply || "").split(/\r?\n/);
  const html = [];
  let paragraph = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    html.push(`<p>${paragraph.map(renderLine).join("<br />")}</p>`);
    paragraph = [];
  };

  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      return;
    }
    if (trimmed === "---") {
      flushParagraph();
      html.push('<hr class="answer-divider" />');
      return;
    }
    if (trimmed.startsWith("**参考文献**")) {
      flushParagraph();
      html.push(`<h4 class="answer-ref-title">${renderInlineMarkdown(trimmed)}</h4>`);
      return;
    }
    if (/^\[\d+\]\s/.test(trimmed)) {
      flushParagraph();
      html.push(`<p class="answer-ref-line">${renderLine(trimmed)}</p>`);
      return;
    }
    paragraph.push(trimmed);
  });
  flushParagraph();
  return html.join("");
}

function scrollToEvidence(index) {
  const el = document.getElementById(`evidence-${index}`);
  if (!el) return;
  el.open = true;
  el.classList.add("evidence-expander--highlight");
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  window.setTimeout(() => el.classList.remove("evidence-expander--highlight"), 1600);
}

function bindCitationLinks(root) {
  root?.querySelectorAll(".cite-ref").forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const n = Number(link.getAttribute("data-cite"));
      if (n) scrollToEvidence(n);
    });
  });
}

function renderAnswerBubble(bubble, reply, contexts, citationCheck) {
  if (!bubble) return;
  bubble.innerHTML = renderAnswerHtml(reply, contexts, citationCheck);
  bindCitationLinks(bubble);
}

function formatDictCounts(obj) {
  if (!obj || typeof obj !== "object" || !Object.keys(obj).length) return "暂无";
  return Object.entries(obj)
    .map(([k, v]) => `${k}: ${v}`)
    .join(" · ");
}

function renderRetrievalExplanation(retrieval, promptVersion) {
  if (!retrievalExpander || !retrievalExpanderBody) return;
  if (!retrieval || typeof retrieval !== "object") {
    retrievalExpander.hidden = true;
    return;
  }
  retrievalExpander.hidden = false;
  const lines = [
    `<p><strong>检索查询：</strong>${escapeHtml(retrieval.rewritten_query || "未生成")}</p>`,
    `<p><strong>返回证据：</strong>${escapeHtml(String(retrieval.retrieved_count ?? 0))} 条</p>`,
    `<p><strong>来源分布：</strong>${escapeHtml(formatDictCounts(retrieval.sources))}</p>`,
    `<p><strong>证据等级：</strong>${escapeHtml(formatDictCounts(retrieval.evidence_levels))}</p>`,
  ];
  if (promptVersion) {
    lines.push(`<p class="retrieval-expander-caption">Prompt 版本：${escapeHtml(promptVersion)}</p>`);
  }
  lines.push(
    '<p class="retrieval-expander-caption">Prompt 栈：query reformulation → grounded system → synthesis → citation validation</p>'
  );
  retrievalExpanderBody.innerHTML = lines.join("");
}

function renderAnswerMeta(data) {
  if (!answerMetaPanel) return;
  const rewritten = data.rewritten_query;
  const check = data.citation_check;
  const promptVersion = data.prompt_version;
  if (!rewritten && !check && !promptVersion) {
    answerMetaPanel.hidden = true;
  } else {
    answerMetaPanel.hidden = false;
    if (rewrittenQueryLine) {
      rewrittenQueryLine.textContent = rewritten ? `改写查询：${rewritten}` : "";
    }
    if (citationCheckLine) {
      if (check && typeof check === "object") {
        const ok = check.ok;
        const label = ok ? "✅ 通过" : "⚠️ 存疑";
        citationCheckLine.textContent = `引用校验：${label}`;
        citationCheckLine.className = "answer-meta-line " + (ok ? "is-ok" : "is-warn");
      } else {
        citationCheckLine.textContent = "";
      }
    }
    if (promptVersionLine) {
      promptVersionLine.textContent = promptVersion ? `Prompt 版本：${promptVersion}` : "";
    }
  }
  renderRetrievalExplanation(data.retrieval, promptVersion);
}

function renderReactSteps(steps) {
  if (!reactStepsEl) return;
  reactStepsEl.innerHTML = "";
  const useReact = useReactEl?.checked !== false;
  const panel = document.getElementById("reactPanel");
  if (panel) panel.style.display = useReact ? "" : "none";
  if (!useReact || !steps || !steps.length) {
    if (reactEmptyEl) reactEmptyEl.style.display = useReact ? "block" : "none";
    return;
  }
  if (reactEmptyEl) reactEmptyEl.style.display = "none";
  steps.forEach((s) => {
    const div = document.createElement("div");
    div.className = "react-step";
    const actionIn = s.action_input ? JSON.stringify(s.action_input, null, 0) : "";
    div.innerHTML = `
      <div><strong>Step ${s.step || "?"}</strong></div>
      ${s.thought ? `<div><strong>Thought:</strong> ${escapeHtml(s.thought)}</div>` : ""}
      ${s.action ? `<div><strong>Action:</strong> ${escapeHtml(s.action)} ${escapeHtml(actionIn)}</div>` : ""}
      ${s.observation ? `<div><strong>Observation:</strong> ${escapeHtml(String(s.observation).slice(0, 600))}${String(s.observation).length > 600 ? "…" : ""}</div>` : ""}
    `;
    reactStepsEl.appendChild(div);
  });
}

function renderEvidence(contexts, citationCheck) {
  if (!evidenceListEl) return;
  evidenceListEl.innerHTML = "";
  const usedSet = new Set((citationCheck?.used_brackets || []).map(Number));
  if (!contexts || !contexts.length) {
    if (evidenceEmptyEl) evidenceEmptyEl.style.display = "block";
    if (evidenceCountEl) evidenceCountEl.textContent = "0 条";
    return;
  }
  if (evidenceEmptyEl) evidenceEmptyEl.style.display = "none";
  if (evidenceCountEl) evidenceCountEl.textContent = `${contexts.length} 条`;
  contexts.forEach((c, i) => {
    const idx = Number(c.index ?? i + 1);
    const title = c.title || "无标题";
    const level = c.evidence_level || "";
    const recordType = c.record_type || "";
    const trialStatus = c.trial_status || "";
    let typeLabel = level;
    if (recordType === "trial_registry") {
      typeLabel = trialStatus ? `试验注册 · ${trialStatus}` : "试验注册";
    }
    const isCited = usedSet.size ? usedSet.has(idx) : true;
    const details = document.createElement("details");
    details.className = "evidence-expander" + (isCited ? " is-cited" : " is-retrieved-only");
    details.id = `evidence-${idx}`;
    details.open = i === 0;
    const summary = document.createElement("summary");
    const badge = isCited ? "已引用" : "仅检索";
    summary.innerHTML = `<span class="evidence-index">[${idx}]</span> ${escapeHtml(title)}${typeLabel ? ` (${escapeHtml(typeLabel)})` : ""} <span class="evidence-tag">${badge}</span>`;
    const body = document.createElement("div");
    body.className = "evidence-expander-body";
    const meta = document.createElement("p");
    meta.className = "evidence-expander-meta";
    const recordHint = recordType === "trial_registry" ? " · 非发表疗效结果" : "";
    meta.textContent = `来源：${c.source || "—"} · 年份：${c.year || "n/a"} · ${c.doc_id || ""}${recordHint}`;
    body.appendChild(meta);
    if (c.url) {
      const link = document.createElement("a");
      link.href = c.url;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = c.url;
      body.appendChild(link);
    }
    const snippet = document.createElement("p");
    snippet.className = "evidence-expander-snippet";
    snippet.textContent = (c.text || c.snippet || "").trim();
    body.appendChild(snippet);
    details.appendChild(summary);
    details.appendChild(body);
    evidenceListEl.appendChild(details);
  });
}

function showWelcome() {
  const msg = TRACK_META[currentTrack].welcome.replace(/\*\*(.+?)\*\*/g, "$1");
  transcript.length = 0;
  transcript.push({ role: "assistant", content: msg });
  if (chatLog) chatLog.innerHTML = "";
  appendBubble("assistant", msg);
  renderReactSteps([]);
  renderEvidence([], null);
  renderAnswerMeta({});
  if (retrievalExpander) retrievalExpander.hidden = true;
}

function setBusy(busy) {
  if (sendBtn) sendBtn.disabled = busy;
  if (chatInput) chatInput.disabled = busy;
}

function getConsultOptions() {
  return {
    top_k: Number(topKSlider?.value || 5),
    use_live_tools: Boolean(useLiveToolsEl?.checked),
    use_react: useReactEl?.checked !== false,
  };
}

async function sendMessage() {
  const text = (chatInput?.value || "").trim();
  if (!text) return;
  chatInput.value = "";
  transcript.push({ role: "user", content: text });
  appendBubble("user", text);
  const pending = appendBubble("assistant", "", { loading: true });
  setBusy(true);

  const opts = getConsultOptions();

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: transcript,
        track: currentTrack,
        use_react: opts.use_react,
        top_k: opts.top_k,
        use_live_tools: opts.use_live_tools,
      }),
    });
    const raw = await resp.text();
    let data = {};
    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch {
        data = { reply: raw };
      }
    }
    if (!resp.ok) {
      let err = data.detail;
      if (Array.isArray(err)) err = err.map((x) => x.msg || JSON.stringify(x)).join("；");
      throw new Error(typeof err === "string" ? err : `HTTP ${resp.status}`);
    }
    const reply = data.reply || "";
    const contexts = data.contexts || data.citations || [];
    transcript.push({ role: "assistant", content: reply });
    const bubble = pending?.querySelector(".consult-bubble");
    if (bubble) {
      bubble.classList.remove("consult-loading");
      if (data.refused) bubble.classList.add("consult-bubble--refused");
      renderAnswerBubble(bubble, reply, contexts, data.citation_check);
    }
    renderReactSteps(data.steps || []);
    renderEvidence(contexts, data.citation_check);
    renderAnswerMeta(data);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (window.showErrorModal) window.showErrorModal(msg);
    const bubble = pending?.querySelector(".consult-bubble");
    if (bubble) {
      bubble.classList.remove("consult-loading");
      bubble.textContent = `发送失败：${msg}`;
    }
  } finally {
    setBusy(false);
    chatLog?.scrollTo(0, chatLog.scrollHeight);
  }
}

topKSlider?.addEventListener("input", () => {
  if (topKValue) topKValue.textContent = topKSlider.value;
});

fillSampleBtn?.addEventListener("click", () => {
  const samples = SAMPLE_QUESTIONS[currentTrack] || [];
  const idx = Number(sampleSelect?.value || 0);
  if (chatInput && samples[idx]) chatInput.value = samples[idx];
});

useReactEl?.addEventListener("change", () => {
  const panel = document.getElementById("reactPanel");
  if (panel) panel.style.display = useReactEl.checked ? "" : "none";
});

trackClinicalBtn?.addEventListener("click", () => {
  if (currentTrack === "clinical") return;
  applyTrackUI("clinical");
  showWelcome();
});
trackNutritionBtn?.addEventListener("click", () => {
  if (currentTrack === "nutrition") return;
  applyTrackUI("nutrition");
  showWelcome();
});

sendBtn?.addEventListener("click", sendMessage);
chatInput?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

applyTrackUI(currentTrack);
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", showWelcome);
} else {
  showWelcome();
}
