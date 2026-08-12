const menuToggle = document.getElementById("menuToggle");
const headerMenuWrap = document.getElementById("headerMenuWrap");

function ensureSiteFavicon() {
  let favicon = document.querySelector("link[rel='icon']");
  if (!favicon) {
    favicon = document.createElement("link");
    favicon.setAttribute("rel", "icon");
    document.head.appendChild(favicon);
  }
  favicon.setAttribute("type", "image/svg+xml");
  favicon.setAttribute("href", "/assets/logo.svg");
}

ensureSiteFavicon();

menuToggle?.addEventListener("click", () => {
  headerMenuWrap?.classList.toggle("is-open");
});

document.querySelectorAll(".menu-item-has-children > a").forEach((link) => {
  link.addEventListener("click", (e) => {
    if (window.innerWidth > 991) return;
    e.preventDefault();
    const li = link.parentElement;
    if (!li) return;
    li.classList.toggle("is-open");
    li.parentElement?.querySelectorAll(".menu-item-has-children").forEach((other) => {
      if (other !== li) other.classList.remove("is-open");
    });
  });
});

window.addEventListener("resize", () => {
  if (window.innerWidth > 991) {
    headerMenuWrap?.classList.remove("is-open");
    document.querySelectorAll(".menu-item-has-children.is-open").forEach((el) => {
      el.classList.remove("is-open");
    });
  }
});

function injectHeaderLogos() {
  document.querySelectorAll(".header-logo a").forEach((brandLink) => {
    if (brandLink.querySelector("img.header-logo-icon")) return;
    const img = document.createElement("img");
    img.className = "header-logo-icon";
    img.src = "/assets/logo.svg";
    img.alt = "Evidence Assistant";
    img.loading = "eager";
    img.decoding = "async";
    brandLink.insertBefore(img, brandLink.firstChild);
  });
}

injectHeaderLogos();

const transitionMask = document.createElement("div");
transitionMask.className = "page-transition-mask";
transitionMask.innerHTML = "<span class='page-transition-text'>Evidence Assistant 正在加载...</span>";
document.body.appendChild(transitionMask);

document.querySelectorAll("a[data-transition='true']").forEach((link) => {
  link.addEventListener("click", (e) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const href = link.getAttribute("href");
    if (!href) return;
    e.preventDefault();
    transitionMask.classList.add("is-active");
    setTimeout(() => {
      window.location.href = href;
    }, 450);
  });
});

function ensureErrorModal() {
  let modal = document.getElementById("errorModal");
  if (modal) return modal;
  modal = document.createElement("div");
  modal.id = "errorModal";
  modal.className = "error-modal";
  modal.innerHTML = `
    <div class="error-modal__panel">
      <h3>操作失败</h3>
      <p id="errorModalText">发生未知错误，请稍后重试。</p>
      <div class="error-modal__actions">
        <button id="errorModalClose" class="btn btn-primary" type="button">我知道了</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  document.getElementById("errorModalClose")?.addEventListener("click", () => {
    modal.classList.remove("is-open");
  });
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.remove("is-open");
  });
  return modal;
}

window.showErrorModal = (message) => {
  const modal = ensureErrorModal();
  const text = document.getElementById("errorModalText");
  if (text) text.textContent = message || "发生未知错误，请稍后重试。";
  modal.classList.add("is-open");
};
