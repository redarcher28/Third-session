(() => {
  "use strict";

  const state = {
    config: null,
    tracks: new Map(),
    selectedTrack: "clinical",
    loading: false,
  };

  const $ = (selector) => document.querySelector(selector);
  const els = {
    apiDot: $("#api-dot"),
    apiStatus: $("#api-status"),
    promptVersion: $("#prompt-version"),
    trackGrid: $("#track-grid"),
    queryTitle: $("#query-title"),
    question: $("#question"),
    queryCounter: $("#query-counter"),
    examples: $("#examples"),
    topK: $("#top-k"),
    liveTools: $("#live-tools"),
    askButton: $("#ask-button"),
    inputNote: $("#input-note"),
    notice: $("#notice"),
    answerSection: $("#answer-section"),
    citationBadge: $("#citation-badge"),
    answerMeta: $("#answer-meta"),
    answer: $("#answer"),
    refusalNote: $("#refusal-note"),
    evidenceCount: $("#evidence-count"),
    railEmpty: $("#rail-empty"),
    evidenceList: $("#evidence-list"),
    retrievalSummary: $("#retrieval-summary"),
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function renderPlainMarkdown(value) {
    return escapeHtml(value)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/^### (.+)$/gm, "<h4>$1</h4>")
      .replace(/^## (.+)$/gm, "<h3>$1</h3>")
      .replace(/^# (.+)$/gm, "<h3>$1</h3>")
      .replace(/\n/g, "<br />");
  }

  function setNotice(message, kind = "info") {
    if (!message) {
      els.notice.hidden = true;
      els.notice.textContent = "";
      return;
    }
    els.notice.hidden = false;
    els.notice.className = `notice notice-${kind}`;
    els.notice.textContent = message;
  }

  function selectedProfile() {
    return state.tracks.get(state.selectedTrack) || null;
  }

  function renderTrackCards() {
    const profiles = Array.from(state.tracks.values());
    els.trackGrid.innerHTML = profiles
      .map((profile, index) => {
        const selected = profile.key === state.selectedTrack;
        const number = String(index + 1).padStart(2, "0");
        const accent = profile.key === "clinical" ? "blue" : "mint";
        return `
          <button class="track-card ${selected ? "is-selected" : ""} track-${accent}"
            type="button" role="radio" aria-checked="${selected}" data-track="${escapeHtml(profile.key)}">
            <span class="track-card-top">
              <span class="track-number">${number}</span>
              <span class="track-icon" aria-hidden="true">${profile.key === "clinical" ? "▤" : "⌕"}</span>
            </span>
            <span class="track-title">${escapeHtml(profile.label.replace(/^赛道[一二] · /, ""))}</span>
            <span class="track-audience">${escapeHtml(profile.audience)}</span>
            <span class="track-description">${escapeHtml(profile.description)}</span>
            <span class="track-contract">
              <span>${escapeHtml(profile.language_contract)}</span>
              <span aria-hidden="true">→</span>
              <span>${escapeHtml(profile.output_contract)}</span>
            </span>
          </button>`;
      })
      .join("");

    els.trackGrid.querySelectorAll("[data-track]").forEach((button) => {
      button.addEventListener("click", () => selectTrack(button.dataset.track));
    });
  }

  function renderExamples() {
    const profile = selectedProfile();
    if (!profile) return;
    els.queryTitle.textContent = profile.key === "clinical"
      ? "把临床问题交给证据库"
      : "把健康问题交给证据库";
    els.inputNote.textContent = profile.forbidden_contract;
    els.examples.innerHTML = profile.sample_questions
      .map((question) => `<button type="button" class="example-chip" data-question="${escapeHtml(question)}">${escapeHtml(question)}</button>`)
      .join("");
    els.examples.querySelectorAll("[data-question]").forEach((chip) => {
      chip.addEventListener("click", () => {
        els.question.value = chip.dataset.question;
        updateCounter();
        els.question.focus();
      });
    });
  }

  function selectTrack(track) {
    if (!state.tracks.has(track)) return;
    state.selectedTrack = track;
    renderTrackCards();
    renderExamples();
    setNotice("");
  }

  function updateCounter() {
    els.queryCounter.textContent = `${els.question.value.length} / 800`;
  }

  function setLoading(loading) {
    state.loading = loading;
    els.askButton.disabled = loading;
    els.askButton.classList.toggle("is-loading", loading);
    els.askButton.querySelector("span:first-child").textContent = loading ? "正在检索与核验…" : "生成带引用回答";
  }

  function renderEvidence(contexts) {
    if (!contexts || contexts.length === 0) {
      els.railEmpty.hidden = false;
      els.evidenceList.innerHTML = "";
      els.evidenceCount.textContent = "0 条";
      return;
    }
    els.railEmpty.hidden = true;
    els.evidenceCount.textContent = `${contexts.length} 条`;
    els.evidenceList.innerHTML = contexts
      .map((context) => {
        const source = context.source || "unknown";
        const year = context.year || "年份未知";
        const href = context.url ? escapeHtml(context.url) : "";
        const title = escapeHtml(context.title || "未命名证据");
        return `
          <article class="evidence-card">
            <div class="evidence-card-top">
              <span class="evidence-index">[${escapeHtml(context.index)}]</span>
              <span class="evidence-level">${escapeHtml(context.evidence_level || "other")}</span>
            </div>
            <h3>${title}</h3>
            <p class="evidence-meta">${escapeHtml(source)} · ${escapeHtml(year)} · ${escapeHtml(context.doc_id || "")}</p>
            <p class="evidence-snippet">${escapeHtml(context.snippet || "暂无摘要片段")}</p>
            ${href ? `<a class="evidence-link" href="${href}" target="_blank" rel="noreferrer">打开原始来源 <span aria-hidden="true">↗</span></a>` : ""}
          </article>`;
      })
      .join("");
  }

  function renderRetrievalSummary(retrieval) {
    if (!retrieval || !retrieval.retrieved_count) {
      els.retrievalSummary.hidden = true;
      return;
    }
    const sources = Object.entries(retrieval.sources || {})
      .map(([key, count]) => `${key} ${count}`)
      .join(" · ");
    els.retrievalSummary.hidden = false;
    els.retrievalSummary.innerHTML = `
      <span class="summary-label">检索记录</span>
      <span>${escapeHtml(retrieval.retrieved_count)} 条证据 · ${escapeHtml(sources || "来源未知")}</span>
      <span class="summary-query">查询：${escapeHtml(retrieval.rewritten_query || "")}</span>`;
  }

  function renderResponse(response) {
    els.answerSection.hidden = false;
    els.answer.innerHTML = renderPlainMarkdown(response.answer || "暂无回答");
    const check = response.citation_check || {};
    const citationOk = check.ok !== false;
    els.citationBadge.className = `badge ${citationOk ? "badge-ok" : "badge-warn"}`;
    els.citationBadge.textContent = citationOk ? "引用已校验" : "引用需复核";
    const used = (check.used_brackets || []).map((number) => `[${number}]`).join(" ");
    els.answerMeta.innerHTML = `
      <span>${response.refused ? "安全拒答" : "基于检索证据生成"}</span>
      <span class="meta-divider">·</span>
      <span>改写查询：${escapeHtml(response.rewritten_query || "未生成")}</span>
      ${used ? `<span class="meta-cites">已使用 ${escapeHtml(used)}</span>` : ""}`;
    if (response.refused) {
      els.refusalNote.hidden = false;
      els.refusalNote.textContent = "当前回答保留了证据边界：没有足够可核对的材料时，系统不会用常识补写结论。";
    } else {
      els.refusalNote.hidden = true;
    }
    renderEvidence(response.contexts || []);
    renderRetrievalSummary(response.retrieval || {});
    els.answerSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function askQuestion() {
    if (state.loading) return;
    const question = els.question.value.trim();
    if (!question) {
      setNotice("先输入一个问题，再开始检索。", "warn");
      els.question.focus();
      return;
    }
    setNotice("");
    setLoading(true);
    try {
      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          track: state.selectedTrack,
          top_k: Number(els.topK.value),
          use_live_tools: els.liveTools.checked,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "问答接口调用失败");
      renderResponse(payload);
    } catch (error) {
      setNotice(error.message || "无法连接到本地后端。", "error");
    } finally {
      setLoading(false);
    }
  }

  async function bootstrap() {
    try {
      const [configResponse, healthResponse] = await Promise.all([
        fetch("/config/tracks"),
        fetch("/health"),
      ]);
      if (!configResponse.ok) throw new Error("赛道配置加载失败");
      state.config = await configResponse.json();
      state.config.tracks.forEach((profile) => state.tracks.set(profile.key, profile));
      state.selectedTrack = state.tracks.has("clinical") ? "clinical" : state.config.tracks[0].key;
      renderTrackCards();
      renderExamples();
      if (healthResponse.ok) {
        const health = await healthResponse.json();
        els.apiDot.classList.add("is-online");
        els.apiStatus.textContent = "本地服务已连接";
        els.promptVersion.textContent = health.prompt_version || state.config.prompt_version;
      }
    } catch (error) {
      els.apiDot.classList.add("is-offline");
      els.apiStatus.textContent = "后端未连接";
      els.promptVersion.textContent = "请启动 uvicorn";
      setNotice(error.message || "无法加载赛道配置，请确认 FastAPI 已启动。", "error");
    }
  }

  els.question.addEventListener("input", updateCounter);
  els.askButton.addEventListener("click", askQuestion);
  els.question.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") askQuestion();
  });
  bootstrap();
})();

