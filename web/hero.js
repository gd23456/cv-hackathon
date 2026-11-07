// A 3D corridor of images this model made. Plays by itself, never captures scrolling or touch,
// and stays still for people who prefer reduced motion.
(async function () {
  const stage = document.getElementById("hero-stage");
  const hero = document.getElementById("hero");
  const go = document.getElementById("hero-go");

  go?.addEventListener("click", (event) => {
    const prompt = document.getElementById("prompt");
    if (!prompt) return;
    event.preventDefault();
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    document.getElementById("form")?.scrollIntoView({ behavior: still ? "auto" : "smooth", block: "start" });
    prompt.focus({ preventScroll: true });
  });

  let images;
  try {
    const response = await fetch("tunnel/images.json");
    images = await response.json();
  } catch {
    return;
  }

  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fine = matchMedia("(pointer: fine)").matches;
  let width = 0, height = 0, mobile = false;

  const cards = images.map((image, i) => {
    const angle = i * 137.5 * (Math.PI / 180);
    const node = document.createElement("figure");
    node.className = "tunnel-card";
    const img = document.createElement("img");
    img.src = image.src;
    img.alt = "";
    img.decoding = "async";
    img.draggable = false;
    const caption = document.createElement("figcaption");
    caption.textContent = image.prompt;
    node.append(img, caption);
    stage.append(node);
    return {
      node, cos: Math.cos(angle), sin: Math.sin(angle),
      rx: 0.3 + (i % 3) * 0.08, ry: 0.28 + (i % 2) * 0.1,
      tilt: (i % 2 === 0 ? 1 : -1) * (3 + (i % 5) * 2), phase: i / images.length,
    };
  });

  function measure() {
    width = hero.clientWidth;
    height = hero.clientHeight;
    mobile = width < 768;
    stage.style.perspective = mobile ? "800px" : "1200px";
    const size = mobile ? Math.max(118, Math.min(170, width * 0.36)) : Math.max(170, Math.min(250, width * 0.17));
    cards.forEach((card) => { card.node.style.width = `${size}px`; });
  }
  measure();
  addEventListener("resize", measure, { passive: true });

  const fade = (p) => (p < 0.18 ? p / 0.18 : p > 0.82 ? Math.max(0, (0.98 - p) / 0.16) : 1);

  function draw(progress, px, py) {
    const far = mobile ? -1200 : -1700, near = mobile ? 260 : 420, pan = mobile ? 0 : 40;
    for (const card of cards) {
      const p = (progress + card.phase) % 1;
      const x = card.cos * width * card.rx - px * pan;
      const y = card.sin * height * card.ry - py * pan;
      card.node.style.opacity = fade(p);
      card.node.style.transform =
        `translate3d(calc(-50% + ${x.toFixed(1)}px), calc(-50% + ${y.toFixed(1)}px), ${(far + (near - far) * p).toFixed(1)}px) rotateZ(${card.tilt}deg)`;
    }
  }

  if (still) {
    draw(0.42, 0, 0);
    return;
  }

  let targetX = 0, targetY = 0, px = 0, py = 0, visible = true, frame = 0, last = performance.now(), travelled = 0;
  if (fine) {
    addEventListener("pointermove", (event) => {
      targetX = (event.clientX / innerWidth - 0.5) * 2;
      targetY = (event.clientY / innerHeight - 0.5) * 2;
    }, { passive: true });
  }

  function tick(now) {
    frame = 0;
    if (!visible || document.hidden) return;
    travelled += Math.min(now - last, 60) / 42000;
    last = now;
    px += (targetX - px) * 0.05;
    py += (targetY - py) * 0.05;
    // Scrolling the page gives a little extra travel, without taking over the scroll.
    draw((travelled + scrollY * 0.0005) % 1, px, py);
    frame = requestAnimationFrame(tick);
  }
  const start = () => {
    if (!frame && visible && !document.hidden) {
      last = performance.now();
      frame = requestAnimationFrame(tick);
    }
  };
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; start(); }).observe(hero);
  document.addEventListener("visibilitychange", start);
  start();
})();
