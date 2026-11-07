// A slowly turning ring of images made by this model. It never captures scrolling,
// and it stays still for people who prefer reduced motion.
(async function () {
  const stage = document.getElementById("hero-stage");
  const go = document.getElementById("hero-go");

  go?.addEventListener("click", (event) => {
    const prompt = document.getElementById("prompt");
    if (!prompt) return;
    event.preventDefault();
    document.getElementById("form")?.scrollIntoView({ block: "center" });
    prompt.focus({ preventScroll: true });
  });

  let images;
  try {
    images = await (await fetch("tunnel/images.json")).json();
  } catch {
    return;
  }

  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const tilts = [-15, -8, 5, 12, -12, 8, -5, 10, -9];
  const cards = images.map((image, i) => {
    const node = document.createElement("div");
    node.className = "hero-card";
    const img = document.createElement("img");
    img.src = image.src;
    img.alt = "";
    img.decoding = "async";
    img.draggable = false;
    node.append(img);
    stage.append(node);
    return { node, angle: (i / images.length) * Math.PI * 2, tilt: tilts[i % tilts.length] };
  });

  let radiusX = 0, radiusY = 0;
  function measure() {
    const small = stage.clientWidth < 620;
    radiusX = small ? Math.min(stage.clientWidth * 0.36, 140) : Math.min(stage.clientWidth * 0.36, 310);
    radiusY = small ? 112 : 150;
  }
  measure();
  addEventListener("resize", measure, { passive: true });

  let mouseX = 0, mouseY = 0, smoothX = 0, smoothY = 0;
  if (matchMedia("(pointer: fine)").matches) {
    addEventListener("pointermove", (e) => {
      mouseX = e.clientX / innerWidth - 0.5;
      mouseY = e.clientY / innerHeight - 0.5;
    }, { passive: true });
  }

  function draw(turn) {
    for (const card of cards) {
      const a = card.angle + turn;
      const depth = (Math.sin(a) + 1) / 2;
      const x = Math.cos(a) * radiusX, y = Math.sin(a) * radiusY;
      card.node.style.zIndex = String(Math.round(depth * 10));
      card.node.style.opacity = String(0.55 + depth * 0.45);
      card.node.style.transform =
        `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) scale(${(0.82 + depth * 0.26).toFixed(3)}) rotateX(${(smoothY * 14).toFixed(2)}deg) rotateY(${(smoothX * 14).toFixed(2)}deg) rotateZ(${card.tilt}deg)`;
    }
  }

  if (still) {
    draw(0.4);
    return;
  }

  let visible = true, frame = 0, turn = 0, last = performance.now();
  function tick(now) {
    frame = 0;
    if (!visible || document.hidden) return;
    turn += Math.min(now - last, 60) * 0.00016;
    last = now;
    smoothX += (mouseX - smoothX) * 0.06;
    smoothY += (mouseY - smoothY) * 0.06;
    draw(turn);
    frame = requestAnimationFrame(tick);
  }
  const start = () => {
    if (!frame && visible && !document.hidden) {
      last = performance.now();
      frame = requestAnimationFrame(tick);
    }
  };
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; start(); }).observe(stage);
  document.addEventListener("visibilitychange", start);
  draw(0);
  start();
})();
