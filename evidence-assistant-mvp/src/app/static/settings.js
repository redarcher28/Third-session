(() => {
  "use strict";

  const els = {
    form: document.querySelector("#connection-form"),
    apiFormat: document.querySelector("#api-format"),
    baseUrl: document.querySelector("#base-url"),
    model: document.querySelector("#model"),
    apiKey: document.querySelector("#api-key"),
    clearApiKey: document.querySelector("#clear-api-key"),
    testButton: document.querySelector("#test-button"),
    resetButton: document.querySelector("#reset-button"),
    statusSummary: document.querySelector("#status-summary"),
    statusSource: document.querySelector("#status-source"),
    keyHint: document.querySelector("#key-hint"),
    formMessage: document.querySelector("#form-message"),
    homeLink: document.querySelector("#home-link"),
  };

  function setMessage(message, kind = "info") {
    els.formMessage.textContent = message || "";
    els.formMessage.className = `form-message form-message-${kind}`;
  }

  async function request(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || "设置请求失败");
    }
    return payload;
  }

  function renderStatus(status) {
    els.apiFormat.value = status.api_format || "responses";
    els.baseUrl.value = status.base_url || "";
    els.model.value = status.model || "";
    els.statusSummary.textContent = status.api_key_configured
      ? `${status.model} · ${status.api_key_hint}`
      : `${status.model} · 未配置令牌`;
    els.statusSource.textContent = status.source === "runtime"
      ? "前端配置"
      : status.source === "env" ? ".env" : "离线模式";
    els.keyHint.textContent = status.api_key_configured
      ? `当前令牌：${status.api_key_hint}；留空即可保持。`
      : "当前没有有效令牌；保存后会切换到运行时配置。";
    if (status.openwebui_url) {
      els.homeLink.href = status.openwebui_url;
    }
  }

  async function loadStatus() {
    try {
      const payload = await request("/api/settings/status", { method: "GET" });
      renderStatus(payload);
    } catch (error) {
      setMessage(error.message || "无法读取当前配置", "error");
    }
  }

  async function saveSettings(event) {
    event.preventDefault();
    setMessage("正在保存…", "info");
    try {
      const payload = await request("/api/settings/update", {
        method: "POST",
        body: JSON.stringify({
          api_format: els.apiFormat.value,
          base_url: els.baseUrl.value.trim(),
          model: els.model.value.trim(),
          api_key: els.apiKey.value || null,
          clear_api_key: els.clearApiKey.checked,
        }),
      });
      els.apiKey.value = "";
      els.clearApiKey.checked = false;
      renderStatus(payload.status);
      setMessage("已保存，新的模型连接已对后续 RAG 请求生效。", "success");
    } catch (error) {
      setMessage(error.message || "保存失败", "error");
    }
  }

  async function testConnection() {
    els.testButton.disabled = true;
    setMessage("正在请求 /v1/models（不会发送聊天内容）…", "info");
    try {
      const result = await request("/api/settings/test", { method: "POST", body: "{}" });
      setMessage(result.message || "测试完成", result.ok ? "success" : "error");
    } catch (error) {
      setMessage(error.message || "测试失败", "error");
    } finally {
      els.testButton.disabled = false;
    }
  }

  async function resetSettings() {
    if (!window.confirm("删除前端保存的覆盖并恢复 .env 配置？")) return;
    els.resetButton.disabled = true;
    try {
      const payload = await request("/api/settings/reset", { method: "POST", body: "{}" });
      renderStatus(payload.status);
      els.apiKey.value = "";
      setMessage("已恢复 .env / 环境变量配置。", "success");
    } catch (error) {
      setMessage(error.message || "恢复失败", "error");
    } finally {
      els.resetButton.disabled = false;
    }
  }

  els.form.addEventListener("submit", saveSettings);
  els.testButton.addEventListener("click", testConnection);
  els.resetButton.addEventListener("click", resetSettings);
  loadStatus();
})();
