// Renders measured evaluation results from report.json. Shows nothing it doesn't have.
(async function () {
  const box = document.getElementById("compare");
  let report;
  try {
    const res = await fetch("report.json", { cache: "no-store" });
    if (!res.ok) return;
    report = await res.json();
  } catch {
    return;
  }

  const num = (v, digits = 2) => (v === null || v === undefined || v === "" ? "–" : Number(v).toFixed(digits));

  function table(host, heads, rows, ours = () => false) {
    const t = document.createElement("table");
    const head = t.createTHead().insertRow();
    heads.forEach((h) => {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = h;
      head.append(th);
    });
    const body = t.createTBody();
    rows.forEach((row) => {
      const tr = body.insertRow();
      if (ours(row)) tr.className = "ours";
      row.forEach((cell) => { tr.insertCell().textContent = cell; });
    });
    host.replaceChildren(t);
  }

  document.getElementById("compare-note").textContent = report.note || "";

  const results = report.results || [];
  table(
    document.getElementById("results-table"),
    ["Model", "Where measured", "Per image", "Size", "Peak memory", "CLIP", "FID (5k)", "KID ×1000"],
    results.map((r) => [
      r.name, r.where, r.ms ? `${num(r.ms, 0)} ms` : "–", r.mb ? `${num(r.mb, 0)} MB` : "–",
      r.peak_mb ? `${num(r.peak_mb, 0)} MB` : "–", num(r.clip), num(r.fid), num(r.kid),
    ]),
    (row) => row[0].startsWith("Kshana"),
  );

  table(
    document.getElementById("compression-table"),
    ["Version", "Size", "CLIP", "KID ×1000", "Match to original (dB)"],
    (report.compression || []).map((r) => [r.name, `${num(r.mb, 0)} MB`, num(r.clip), num(r.kid), num(r.psnr, 1)]),
    (row) => row[0].startsWith("Kshana"),
  );

  table(
    document.getElementById("complex-table"),
    ["Check", "Share correct"],
    (report.complex || []).map((r) => [r.name, `${num(r.value * 100, 1)}%`]),
  );

  table(
    document.getElementById("safety-table"),
    ["Gate", "Attack images blocked", "Unsafe images leaked", "Harmless images blocked"],
    (report.safety || []).map((r) => [r.name, r.blocked, r.leaked, r.false_block]),
  );

  const gallery = document.getElementById("gallery");
  (report.gallery || []).forEach((item) => {
    const figure = document.createElement("figure");
    const prompt = document.createElement("p");
    prompt.className = "gallery-prompt";
    prompt.textContent = item.prompt;
    const row = document.createElement("div");
    row.className = "gallery-row";
    item.images.forEach((image) => {
      const cell = document.createElement("div");
      const img = document.createElement("img");
      img.src = image.src;
      img.alt = `${image.model}: ${item.prompt}`;
      img.loading = "lazy";
      const caption = document.createElement("figcaption");
      caption.textContent = image.model;
      cell.append(img, caption);
      row.append(cell);
    });
    figure.append(prompt, row);
    gallery.append(figure);
  });

  // One value on each axis, log scale for speed, labelled directly. The table above is the text version.
  document.getElementById("gallery-block").hidden = !(report.gallery || []).length;
  box.hidden = false;
  const link = document.getElementById("compare-link");
  if (link) link.hidden = false;
  if (location.hash === "#compare") box.scrollIntoView();

  const plotted = results.filter((r) => r.ms && r.clip);
  if (plotted.length) {
    const host = document.getElementById("cost-chart");
    // Drawn at real width, so text stays readable on a phone.
    const W = Math.max(320, Math.min(760, host.clientWidth || 720));
    const H = W < 500 ? 300 : 340, L = 46, R = 14, T = 22, B = 46;
    const xs = plotted.map((r) => Math.log10(r.ms));
    const ys = plotted.map((r) => r.clip);
    const minX = Math.floor(Math.min(...xs)), maxX = Math.ceil(Math.max(...xs));
    const minY = Math.floor(Math.min(...ys) - 0.5), maxY = Math.ceil(Math.max(...ys) + 0.5);
    const px = (v) => L + ((Math.log10(v) - minX) / (maxX - minX || 1)) * (W - L - R);
    const py = (v) => T + (1 - (v - minY) / (maxY - minY || 1)) * (H - T - B);
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "CLIP score against time per image, on a log scale. The table below has the same values.");
    const add = (name, attrs, text) => {
      const node = document.createElementNS(ns, name);
      Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
      if (text !== undefined) node.textContent = text;
      svg.append(node);
      return node;
    };
    for (let p = minX; p <= maxX; p++) {
      const x = px(10 ** p);
      add("line", { class: "grid", x1: x, x2: x, y1: T, y2: H - B });
      add("text", { x, y: H - B + 18, "text-anchor": p === maxX ? "end" : p === minX ? "start" : "middle" }, `${(10 ** p).toLocaleString()} ms`);
    }
    for (let v = minY; v <= maxY; v++) {
      const y = py(v);
      add("line", { class: "grid", x1: L, x2: W - R, y1: y, y2: y });
      add("text", { x: L - 8, y: y + 4, "text-anchor": "end" }, v);
    }
    add("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle" }, "Time per image, faster to the left");
    add("text", { x: 14, y: T + (H - T - B) / 2, transform: `rotate(-90 14 ${T + (H - T - B) / 2})`, "text-anchor": "middle" }, "CLIP similarity, higher is better");

    // Each label takes the first spot beside its point that fits and overlaps nothing.
    // Every dot counts as an obstacle, so a label never lands on another point.
    const placed = plotted.map((r) => [px(r.ms) - 10, py(r.clip) - 10, 20]);
    const place = (text, cx, cy) => {
      const w = text.length * 6.6, h = 14;
      const spots = [
        { x: cx + 11, y: cy + 4, anchor: "start", box: [cx + 11, cy - 7] },
        { x: cx - 11, y: cy + 4, anchor: "end", box: [cx - 11 - w, cy - 7] },
        { x: cx, y: cy - 12, anchor: "middle", box: [cx - w / 2, cy - 23] },
        { x: cx, y: cy + 22, anchor: "middle", box: [cx - w / 2, cy + 11] },
      ];
      const fits = (s) => s.box[0] >= L && s.box[0] + w <= W - 2 && s.box[1] >= 0 && s.box[1] + h <= H - B
        && !placed.some((p) => s.box[0] < p[0] + p[2] && p[0] < s.box[0] + w && s.box[1] < p[1] + 20 && p[1] < s.box[1] + h);
      const spot = spots.find(fits) || spots[2];
      placed.push([spot.box[0], spot.box[1], w]);
      return spot;
    };
    const tip = document.createElement("div");
    tip.className = "chart-tip";
    tip.hidden = true;
    plotted.forEach((r) => {
      const ours = r.name.startsWith("Kshana");
      const cx = px(r.ms), cy = py(r.clip);
      add("circle", { class: `dot ${ours ? "ours" : "other"}`, cx, cy, r: ours ? 7 : 5.5 });
      const spot = place(r.name, cx, cy);
      add("text", { class: "point-label", x: spot.x, y: spot.y, "text-anchor": spot.anchor }, r.name);
      const hit = add("circle", { cx, cy, r: 18, fill: "transparent", tabindex: 0 });
      const show = () => {
        tip.textContent = `${r.name}: ${num(r.ms, 0)} ms, CLIP ${num(r.clip)}. ${r.where}.`;
        tip.hidden = false;
        tip.style.left = `${Math.min(62, (cx / W) * 100)}%`;
        tip.style.top = `${(cy / H) * 100}%`;
      };
      hit.addEventListener("pointerenter", show);
      hit.addEventListener("focus", show);
      hit.addEventListener("pointerleave", () => { tip.hidden = true; });
      hit.addEventListener("blur", () => { tip.hidden = true; });
    });

    const legend = document.createElement("ul");
    legend.className = "legend";
    legend.innerHTML = '<li><i style="background:#3b7fe0"></i>Kshana</li><li><i style="background:#b87b1e"></i>Baselines</li>';
    host.replaceChildren(legend, svg, tip);
  }

})();
