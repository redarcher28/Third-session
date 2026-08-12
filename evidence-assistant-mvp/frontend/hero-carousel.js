/**
 * 首页 3D 海报轮播：自动播放 + 箭头/圆点 + 键盘 + 触摸滑动
 */
(function initHeroCarousel() {
  const wrap = document.querySelector(".hero-carousel-wrap");
  if (!wrap) return;

  const track = wrap.querySelector(".hero-carousel-track");
  const slides = Array.from(wrap.querySelectorAll(".hero-poster-slide"));
  const prevBtn = wrap.querySelector(".hero-carousel-arrow--prev");
  const nextBtn = wrap.querySelector(".hero-carousel-arrow--next");
  const dotsContainer = wrap.querySelector(".hero-carousel-dots");

  if (!track || slides.length === 0) return;

  let current = 0;
  let timer = null;
  const INTERVAL = 5500;
  let touchStartX = 0;

  slides.forEach((_, i) => {
    const dot = document.createElement("button");
    dot.type = "button";
    dot.className = "hero-carousel-dot" + (i === 0 ? " is-active" : "");
    dot.setAttribute("aria-label", `第 ${i + 1} 张海报`);
    dot.addEventListener("click", () => goTo(i, true));
    dotsContainer?.appendChild(dot);
  });

  const dots = dotsContainer ? Array.from(dotsContainer.querySelectorAll(".hero-carousel-dot")) : [];

  function layout() {
    const n = slides.length;
    slides.forEach((slide, i) => {
      let offset = i - current;
      if (offset > n / 2) offset -= n;
      if (offset < -n / 2) offset += n;

      const isActive = offset === 0;
      const abs = Math.abs(offset);

      slide.classList.toggle("is-active", isActive);
      slide.style.zIndex = String(20 - abs);
      slide.style.opacity = abs > 1 ? "0" : isActive ? "1" : "0.45";
      slide.style.filter = isActive ? "none" : "brightness(0.72) blur(1px)";

      const tx = offset * 108;
      const scale = isActive ? 1 : 0.84;
      const rotY = offset * -22;
      const tz = isActive ? 0 : -180 - abs * 40;

      slide.style.transform = `
        translateX(${tx}%)
        scale(${scale})
        rotateY(${rotY}deg)
        translateZ(${tz}px)
      `;
    });

    dots.forEach((dot, i) => dot.classList.toggle("is-active", i === current));
  }

  function goTo(index, userTriggered) {
    const n = slides.length;
    current = ((index % n) + n) % n;
    layout();
    if (userTriggered) restartAutoplay();
  }

  function next(userTriggered) {
    goTo(current + 1, userTriggered);
  }

  function prev(userTriggered) {
    goTo(current - 1, userTriggered);
  }

  function restartAutoplay() {
    clearInterval(timer);
    timer = setInterval(() => next(false), INTERVAL);
  }

  prevBtn?.addEventListener("click", () => prev(true));
  nextBtn?.addEventListener("click", () => next(true));

  wrap.addEventListener("mouseenter", () => clearInterval(timer));
  wrap.addEventListener("mouseleave", restartAutoplay);

  wrap.addEventListener(
    "touchstart",
    (e) => {
      touchStartX = e.changedTouches[0]?.clientX ?? 0;
    },
    { passive: true }
  );

  wrap.addEventListener(
    "touchend",
    (e) => {
      const dx = (e.changedTouches[0]?.clientX ?? 0) - touchStartX;
      if (Math.abs(dx) < 40) return;
      if (dx < 0) next(true);
      else prev(true);
    },
    { passive: true }
  );

  document.addEventListener("keydown", (e) => {
    if (!wrap.matches(":hover") && document.activeElement?.tagName !== "BODY") return;
    if (e.key === "ArrowLeft") prev(true);
    if (e.key === "ArrowRight") next(true);
  });

  layout();
  restartAutoplay();
})();
