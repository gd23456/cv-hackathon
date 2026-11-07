// Draws panels as folders: a clip path and a bright rim fitted to each panel's real size.
// CSS sets the tab with unitless custom properties, so it can change at each breakpoint.
// Without this script, panels are still usable rounded rectangles.
window.Folder = (function () {
  const SVG = "http://www.w3.org/2000/svg";

  function folderPath(w, h, tabX, tabW, tabH, r) {
    const shoulder = tabH * 1.35;
    const end = tabX + tabW;
    const bottom = `L${w} ${h - r}A${r} ${r} 0 0 1 ${w - r} ${h}L${r} ${h}A${r} ${r} 0 0 1 0 ${h - r}Z`;
    const riseLeft = tabX <= 0
      ? `M0 ${r}A${r} ${r} 0 0 1 ${r} 0`
      : `M0 ${tabH + r}A${r} ${r} 0 0 1 ${r} ${tabH}L${tabX} ${tabH}C${tabX + shoulder / 2} ${tabH} ${tabX + shoulder / 2} 0 ${tabX + shoulder} 0`;
    const fallRight = end >= w
      ? `L${w - r} 0A${r} ${r} 0 0 1 ${w} ${r}`
      : `L${end - shoulder} 0C${end - shoulder / 2} 0 ${end - shoulder / 2} ${tabH} ${end} ${tabH}L${w - r} ${tabH}A${r} ${r} 0 0 1 ${w} ${tabH + r}`;
    return riseLeft + fallRight + bottom;
  }

  function attach(node) {
    const rim = document.createElementNS(SVG, "svg");
    rim.setAttribute("class", "gf-rim");
    rim.setAttribute("aria-hidden", "true");
    const line = document.createElementNS(SVG, "path");
    rim.append(line);
    node.prepend(rim);

    const draw = () => {
      const w = node.offsetWidth;
      const h = node.offsetHeight;
      if (!w || !h) return;
      const css = getComputedStyle(node);
      const read = (name, fallback) => {
        const value = parseFloat(css.getPropertyValue(name));
        return Number.isFinite(value) ? value : fallback;
      };
      const r = read("--r", 22);
      const tabH = read("--tab-h", 30);
      const tabW = Math.min(read("--tab-w", 170), w - r);
      const slots = read("--tab-slots", 1);
      let tabX = 0;
      if (css.getPropertyValue("--tab-x").trim() === "right") tabX = w - tabW;
      else if (read("--tab-spread", 0) && slots > 1) tabX = (read("--i", 0) / (slots - 1)) * (w - tabW - r);
      const d = folderPath(w, h, tabX, tabW, tabH, r);
      node.style.clipPath = `path('${d}')`;
      node.style.setProperty("--tab-text-x", `${tabX > 0 ? tabX + tabH * 1.35 : 20}px`);
      rim.setAttribute("viewBox", `0 0 ${w} ${h}`);
      rim.style.width = `${w}px`;
      rim.style.height = `${h}px`;
      line.setAttribute("d", d);
    };
    new ResizeObserver(draw).observe(node);
    draw();
  }

  // Light and tilt follow the pointer. Writes custom properties only, so nothing re-renders.
  function tilt(node) {
    if (!matchMedia("(pointer: fine) and (prefers-reduced-motion: no-preference)").matches) return;
    let frame = 0, px = 0, py = 0;
    const write = () => {
      frame = 0;
      node.style.setProperty("--px", px.toFixed(3));
      node.style.setProperty("--py", py.toFixed(3));
    };
    const queue = () => { if (!frame) frame = requestAnimationFrame(write); };
    node.addEventListener("pointermove", (event) => {
      const box = node.getBoundingClientRect();
      px = ((event.clientX - box.left) / box.width) * 2 - 1;
      py = ((event.clientY - box.top) / box.height) * 2 - 1;
      queue();
    });
    node.addEventListener("pointerleave", () => { px = 0; py = 0; queue(); });
  }

  document.querySelectorAll(".gf, .gf-shape").forEach(attach);
  document.querySelectorAll("[data-tilt]").forEach(tilt);
  return { attach };
})();
