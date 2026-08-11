(() => {
  "use strict";

  const BRIDGE_SELECTOR = "[data-evidence-model-bridge]";
  const PRESETS = Object.freeze({
    custom: {
      label: "自定义兼容接口",
      note: "可手动填写任何 OpenAI-compatible 服务。",
    },
    "byeapi-luna": {
      label: "ByeAPI · GPT-5.6 Luna",
      apiFormat: "responses",
      baseUrl: "https://api.byeapi.top",
      model: "gpt-5.6-luna",
      note: "Responses 接口；保留当前 Luna 配置，不开放额外推理强度调节。",
    },
    "deepseek-flash": {
      label: "DeepSeek · V4 Flash（推荐）",
      apiFormat: "openai",
      baseUrl: "https://api.deepseek.com",
      model: "deepseek-v4-flash",
      note: "OpenAI-compatible；默认使用本地向量，避免向 DeepSeek 请求 Embedding。",
    },
    "deepseek-pro": {
      label: "DeepSeek · V4 Pro",
      apiFormat: "openai",
      baseUrl: "https://api.deepseek.com",
      model: "deepseek-v4-pro",
      note: "OpenAI-compatible；适合更复杂的证据综合，但响应时间和消耗可能更高。",
    },
  });

  const configuredBackendUrl = "__EVIDENCE_BACKEND_URL_VALUE__";
  const backendUrl = String(
    window.__EVIDENCE_BACKEND_URL__ || (
      configuredBackendUrl === "__EVIDENCE_BACKEND_URL_VALUE__"
        ? "http://127.0.0.1:8000"
        : configuredBackendUrl
    ),
  ).replace(/\/+$/, "");

  function normalizeBaseUrl(value) {
    return String(value || "").trim().replace(/\/+$/, "");
  }

  function identifyPreset(status) {
    const baseUrl = normalizeBaseUrl(status.base_url);
    return Object.entries(PRESETS).find(([key, preset]) => (
      key !== "custom" &&
      preset.apiFormat === status.api_format &&
      normalizeBaseUrl(preset.baseUrl) === baseUrl &&
      preset.model === status.model
    ))?.[0] || "custom";
  }

  async function request(path, options = {}) {
    const response = await fetch(`${backendUrl}${path}`, {
      ...options,
      mode: "cors",
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_error) {
      payload = null;
    }
    if (!response.ok) {
      throw new Error(payload?.detail || `请求失败（HTTP ${response.status}）`);
    }
    return payload;
  }

  function setMessage(elements, message, kind = "info") {
    elements.message.textContent = message || "";
    elements.message.dataset.kind = kind;
  }

  function renderPresetNote(elements, key) {
    const preset = PRESETS[key] || PRESETS.custom;
    elements.presetNote.textContent = preset.note;
    elements.apiFormatHint.textContent = preset.apiFormat === "openai"
      ? "DeepSeek 使用 OpenAI Chat Completions 兼容协议。"
      : preset.apiFormat === "responses"
        ? "ByeAPI / Codex 配置请选择 Responses。"
        : "按服务商提供的 Messages 协议填写。";
  }

  function applyPreset(elements, key, announce = true) {
    const preset = PRESETS[key] || PRESETS.custom;
    elements.providerPreset.value = key;
    renderPresetNote(elements, key);
    if (key === "custom") {
      if (announce) setMessage(elements, "已切换到自定义接口，可手动填写连接信息。", "info");
      return;
    }
    elements.apiFormat.value = preset.apiFormat;
    elements.baseUrl.value = preset.baseUrl;
    elements.model.value = preset.model;
    if (announce) setMessage(elements, `已套用${preset.label}预设，请填写或确认 API Key。`, "info");
  }

  function renderStatus(elements, status) {
    const sourceLabel = { env: ".env", runtime: "本机覆盖", none: "未配置" }[status.source]
      || status.source
      || "—";
    const configured = Boolean(status.api_key_configured);
    elements.status.textContent = configured
      ? `${status.model} · ${status.api_key_hint}`
      : `${status.model} · 未配置令牌`;
    elements.status.dataset.state = configured ? "ready" : "idle";
    elements.statusSource.textContent = `${sourceLabel} · ${status.api_format}`;
    elements.keyHint.textContent = configured
      ? `当前令牌：${status.api_key_hint}；留空保存时会保留它。`
      : "尚未配置有效令牌；令牌不会回显或写入 Git。";
    elements.apiFormat.value = status.api_format || "responses";
    elements.baseUrl.value = status.base_url || "";
    elements.model.value = status.model || "";
    applyPreset(elements, identifyPreset(status), false);
  }

  function createBridge() {
    const shell = document.createElement("section");
    shell.className = "evidence-bridge";
    shell.dataset.evidenceModelBridge = "true";
    shell.innerHTML = `
      <div class="evidence-bridge__head">
        <div class="evidence-bridge__identity">
          <span class="evidence-bridge__mark" aria-hidden="true"><i></i><i></i><i></i></span>
          <div>
            <p class="evidence-bridge__eyebrow">EVIDENCE DESK / MODEL</p>
            <h3 class="evidence-bridge__title">证据模型连接</h3>
            <p class="evidence-bridge__description">
              在 Open WebUI 设置里切换模型；赛道 Prompt、RAG 检索与引用规则仍由后端统一控制。
            </p>
          </div>
        </div>
        <div class="evidence-bridge__status-stack">
          <span class="evidence-bridge__status" data-eb-status data-state="idle">读取中…</span>
          <span class="evidence-bridge__source" data-eb-source>—</span>
        </div>
      </div>
      <form data-eb-form>
        <div class="evidence-bridge__grid">
          <label class="evidence-bridge__field">
            <span>服务商预设</span>
            <select data-eb-provider aria-label="证据模型服务商预设">
              <option value="custom">自定义兼容接口</option>
              <option value="byeapi-luna">ByeAPI · GPT-5.6 Luna</option>
              <option value="deepseek-flash">DeepSeek · V4 Flash（推荐）</option>
              <option value="deepseek-pro">DeepSeek · V4 Pro</option>
            </select>
            <small data-eb-preset-note>可手动填写任何 OpenAI-compatible 服务。</small>
          </label>
          <label class="evidence-bridge__field">
            <span>接口协议</span>
            <select data-eb-format aria-label="证据模型接口协议">
              <option value="responses">OpenAI Responses</option>
              <option value="openai">OpenAI Chat Completions</option>
              <option value="anthropic">Anthropic Messages</option>
            </select>
            <small data-eb-format-hint>按服务商提供的协议填写。</small>
          </label>
          <label class="evidence-bridge__field evidence-bridge__field--wide">
            <span>服务地址</span>
            <input data-eb-base-url type="url" autocomplete="url" placeholder="https://api.byeapi.top" />
            <small>DeepSeek 使用官方根地址；其他服务可填根地址或带 /v1 的地址。</small>
          </label>
          <label class="evidence-bridge__field">
            <span>模型 ID</span>
            <input data-eb-model autocomplete="off" placeholder="gpt-5.6-luna" />
            <small>以服务商模型列表中的 ID 为准。</small>
          </label>
          <label class="evidence-bridge__field">
            <span>API Key / Token</span>
            <input data-eb-key type="password" maxlength="1000" autocomplete="new-password" placeholder="留空保持当前令牌" />
            <small data-eb-key-hint>令牌不会回显到页面或提交到 Git。</small>
          </label>
        </div>
        <div class="evidence-bridge__policy" role="note">
          <span class="evidence-bridge__check" aria-hidden="true">✓</span>
          <div><strong>赛道策略已固定</strong><p>不开放 reasoning effort、temperature、top_p、max tokens 等细调，保证赛道一/二的演示口径一致。</p></div>
        </div>
        <label class="evidence-bridge__clear"><input data-eb-clear type="checkbox" /> 清除当前保存的令牌</label>
        <div class="evidence-bridge__actions">
          <button class="evidence-bridge__button evidence-bridge__button--primary" data-eb-save type="submit">保存连接</button>
          <button class="evidence-bridge__button evidence-bridge__button--secondary" data-eb-test type="button">测试模型列表</button>
          <button class="evidence-bridge__button evidence-bridge__button--quiet" data-eb-reset type="button">恢复 .env 配置</button>
          <p class="evidence-bridge__message" data-eb-message role="status" aria-live="polite"></p>
        </div>
      </form>
    `;

    const elements = {
      shell,
      form: shell.querySelector("[data-eb-form]"),
      providerPreset: shell.querySelector("[data-eb-provider]"),
      presetNote: shell.querySelector("[data-eb-preset-note]"),
      apiFormat: shell.querySelector("[data-eb-format]"),
      apiFormatHint: shell.querySelector("[data-eb-format-hint]"),
      baseUrl: shell.querySelector("[data-eb-base-url]"),
      model: shell.querySelector("[data-eb-model]"),
      apiKey: shell.querySelector("[data-eb-key]"),
      keyHint: shell.querySelector("[data-eb-key-hint]"),
      clearApiKey: shell.querySelector("[data-eb-clear]"),
      saveButton: shell.querySelector("[data-eb-save]"),
      testButton: shell.querySelector("[data-eb-test]"),
      resetButton: shell.querySelector("[data-eb-reset]"),
      status: shell.querySelector("[data-eb-status]"),
      statusSource: shell.querySelector("[data-eb-source]"),
      message: shell.querySelector("[data-eb-message]"),
    };

    elements.providerPreset.addEventListener("change", () => {
      applyPreset(elements, elements.providerPreset.value);
    });

    elements.form.addEventListener("submit", async (event) => {
      event.preventDefault();
      elements.saveButton.disabled = true;
      setMessage(elements, "正在保存…", "info");
      try {
        const payload = await request("/api/settings/update", {
          method: "POST",
          body: JSON.stringify({
            api_format: elements.apiFormat.value,
            base_url: elements.baseUrl.value.trim(),
            model: elements.model.value.trim(),
            api_key: elements.apiKey.value || null,
            clear_api_key: elements.clearApiKey.checked,
          }),
        });
        elements.apiKey.value = "";
        elements.clearApiKey.checked = false;
        renderStatus(elements, payload.status);
        setMessage(elements, "已保存，新的模型连接已对后续 RAG 请求生效。", "success");
      } catch (error) {
        setMessage(elements, error.message || "保存失败", "error");
      } finally {
        elements.saveButton.disabled = false;
      }
    });

    elements.testButton.addEventListener("click", async () => {
      elements.testButton.disabled = true;
      setMessage(elements, "正在请求模型列表（不会发送聊天内容）…", "info");
      try {
        const result = await request("/api/settings/test", { method: "POST", body: "{}" });
        setMessage(elements, result.message || "测试完成", result.ok ? "success" : "error");
      } catch (error) {
        setMessage(elements, error.message || "测试失败", "error");
      } finally {
        elements.testButton.disabled = false;
      }
    });

    elements.resetButton.addEventListener("click", async () => {
      if (!window.confirm("删除前端保存的覆盖并恢复 .env 配置？")) return;
      elements.resetButton.disabled = true;
      setMessage(elements, "正在恢复 .env 配置…", "info");
      try {
        const payload = await request("/api/settings/reset", { method: "POST", body: "{}" });
        renderStatus(elements, payload.status);
        elements.apiKey.value = "";
        setMessage(elements, "已恢复 .env / 环境变量配置。", "success");
      } catch (error) {
        setMessage(elements, error.message || "恢复失败", "error");
      } finally {
        elements.resetButton.disabled = false;
      }
    });

    request("/api/settings/status")
      .then((payload) => renderStatus(elements, payload))
      .catch((error) => setMessage(elements, error.message || "无法读取当前配置", "error"));

    return shell;
  }

  function findGeneralScroller() {
    const panel = document.querySelector("#tab-general");
    if (!panel) return null;
    return [...panel.children].find((child) => (
      child.classList.contains("flex-1") && child.classList.contains("overflow-y-auto")
    )) || null;
  }

  function attachBridge() {
    const scroller = findGeneralScroller();
    if (!scroller || scroller.querySelector(BRIDGE_SELECTOR)) return;
    scroller.prepend(createBridge());
  }

  function boot() {
    if (!document.body) return;
    const observer = new MutationObserver(attachBridge);
    observer.observe(document.body, { childList: true, subtree: true });
    attachBridge();
    window.setInterval(attachBridge, 1200);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
