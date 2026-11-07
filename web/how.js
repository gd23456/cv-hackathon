// "How it finds the best image": one recorded search run, as cards that stack while you scroll.
// Scores and checks come from the saved run in search/story.json; nothing here is typed in by hand except the wording.
(async function () {
  const box = document.getElementById("how");
  const stack = document.getElementById("how-stack");
  const note = document.getElementById("how-note");
  if (!box || !stack) return;

  const load = async (path) => {
    const response = await fetch(path);
    if (!response.ok) throw new Error(path);
    return response.json();
  };
  let story, benefit;
  try {
    [story, benefit] = await Promise.all([load("search/story.json"), load("search/benefit.json")]);
  } catch {
    return;
  }

  // The images the run produced that this section stops on, and what each one shows.
  const STOPS = [
    { n: 1, title: "The first image appears right away", alt: "Two blue cups with red apples on top and to the right.",
      text: `Both apple checks pass, but the detector finds no cup, so colour and position can't be checked yet. It's the only image so far, so you see it after ${(story.timings.first_image_ms / 1000).toFixed(1)} s.` },
    { n: 3, title: "The wrong colour gets caught", alt: "A red mug holding an apple, with a smaller apple to its right.",
      text: "The cup came out red. The detector finds it, the colour check measures almost no blue, and the position check fails too. Nobody ever sees this image." },
    { n: 6, title: "The checks can be wrong", alt: "A blue cup with a red apple on its left and a tiny apple on its right.",
      text: "This looks close, but the detector misses the cup, so the cup check fails anyway. The score rises only because the CLIP match is higher." },
    { n: 9, title: "Cup found, wrong side", alt: "Two joined blue cups with a red apple at the bottom right.",
      text: "With the cup found, its colour and position can finally be checked. The cup is blue, but the apple is on the wrong side." },
    { n: 10, title: "Every check passes", alt: "A red apple on the left and a blue cup on the right, on a wooden table.", final: true,
      text: `Red apple on the left, blue cup on the right. This try used a reworded prompt, but it was scored against your original words. It came on the last try, at ${(story.timings.compute_ms / 1000).toFixed(1)} s, so a shorter budget would have missed it.` },
  ];

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const score = (value) => value.toFixed(2);
  const attempts = story.attempts;
  const total = attempts.length;
  const finalAttempt = attempts.filter((a) => a.new_best).at(-1);

  // Best score before each attempt, worked out from the run itself.
  let running = null;
  const bestBefore = new Map();
  for (const attempt of attempts) {
    bestBefore.set(attempt.n, running);
    if (attempt.new_best) running = attempt.score;
  }

  const image = (attempt, alt) => {
    const img = el("img");
    img.src = attempt.src;
    img.alt = alt;
    img.loading = "lazy";
    img.decoding = "async";
    img.width = 512;
    img.height = 512;
    return img;
  };

  const MARK = { pass: "#i-pass", fail: "#i-fail", not_verifiable: "#i-none" };
  const checkRow = (item) => {
    const row = el("li", "how-check");
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", MARK[item.status]);
    svg.append(use);
    const words = { pass: "passed", fail: "failed", not_verifiable: "couldn't be checked" };
    const label = el("span", "how-check-label", item.label);
    label.append(el("span", "sr-only", `, ${words[item.status]}`));
    row.append(svg, label, el("span", "how-check-detail", item.detail));
    return row;
  };

  const step = (index, card, tab) => {
    const item = el("li", "how-step");
    item.style.setProperty("--i", String(index));
    card.classList.add("gf-shape");
    const label = el("span", "gf-tab", tab);
    label.setAttribute("aria-hidden", "true");
    card.prepend(label);
    item.append(card);
    return item;
  };

  const verdict = (attempt, isFinal) => {
    const before = bestBefore.get(attempt.n);
    const kind = isFinal ? "final" : attempt.new_best ? "best" : "dropped";
    const word = { final: "Final image", best: "New best", dropped: "Not kept" }[kind];
    const detail = before === null ? `score ${score(attempt.score)}`
      : attempt.new_best ? `score ${score(attempt.score)}, up from ${score(before)}`
      : `score ${score(attempt.score)}, best so far ${score(before)}`;
    return el("p", `how-verdict how-verdict-${kind}`, `${word}: ${detail}`);
  };

  function promptCard() {
    const card = el("article", "how-card how-card-intro");
    const mosaic = el("div", "how-mosaic");
    mosaic.setAttribute("aria-hidden", "true");
    attempts.filter((a) => a.src).forEach((a) => mosaic.append(image(a, "")));
    const copy = el("div", "how-intro");
    copy.append(el("p", "how-kicker", "The prompt"));
    copy.append(el("h3", "how-intro-title", "One prompt becomes a checklist"));
    copy.append(el("p", "how-prompt", `“${story.prompt}”`));
    const chips = el("ul", "how-chips");
    attempts.find((a) => a.checks).checks
      .filter((c) => c.kind !== "alignment")
      .forEach((c) => chips.append(el("li", "how-chip", c.label)));
    copy.append(chips);
    copy.append(el("p", "how-text", `The server then makes image after image from new noise until its ${story.plan.budget_s.toFixed(0)} s budget runs out, about ${(story.plan.attempt_ms / 1000).toFixed(1)} s each. Every image goes through the safety and quality checks before it's scored.`));
    card.append(mosaic, copy);
    return card;
  }

  function attemptCard(stop) {
    const attempt = attempts.find((a) => a.n === stop.n);
    const card = el("article", "how-card how-card-attempt");
    const visual = el("div", "how-visual");
    visual.append(image(attempt, stop.alt));
    const copy = el("div", "how-copy");
    copy.append(el("p", "how-kicker", `Image ${attempt.n} of ${total}`));
    copy.append(el("h3", "how-title", stop.title));
    copy.append(el("p", "how-text", stop.text));
    copy.append(verdict(attempt, Boolean(stop.final)));
    const checks = el("ul", "how-checks");
    attempt.checks.filter((c) => c.kind !== "alignment").forEach((c) => checks.append(checkRow(c)));
    copy.append(checks);
    const clip = attempt.checks.find((c) => c.kind === "alignment");
    if (clip && clip.value !== undefined) copy.append(el("p", "how-clip", `CLIP match to the prompt: ${clip.value.toFixed(2)}`));
    card.append(visual, copy);
    return card;
  }

  function sheetCard() {
    const card = el("article", "how-card how-card-sheet");
    const head = el("div", "how-sheet-head");
    head.append(el("p", "how-kicker", `All ${total} tries`));
    head.append(el("h3", "how-title", "Only a higher score replaces the image"));
    head.append(el("p", "how-text", `Each score is 60% checks passed and 40% CLIP match to the prompt. Best so far: ${attempts.filter((a) => a.new_best).map((a) => score(a.score)).join(" → ")}. Images the safety or quality checks reject are never kept; this run had none.`));
    const grid = el("ol", "how-sheet");
    attempts.forEach((attempt) => {
      const kind = attempt === finalAttempt ? "final" : attempt.new_best ? "best" : "dropped";
      const tile = el("li", `how-tile how-tile-${kind}`);
      const passed = attempt.checks.filter((c) => c.status === "pass").length;
      const scored = attempt.checks.filter((c) => c.kind !== "alignment").length;
      tile.append(image(attempt, `Image ${attempt.n}: score ${score(attempt.score)}, ${passed} of ${scored} checks passed.`));
      tile.append(el("span", "how-tile-score", score(attempt.score)));
      grid.append(tile);
    });
    const legend = el("ul", "how-legend");
    [["best", "New best"], ["final", "Final image"]].forEach(([kind, label]) => {
      const entry = el("li", `how-legend-${kind}`, label);
      legend.append(entry);
    });
    head.append(legend);
    card.append(head, grid);
    return card;
  }

  function benefitCard() {
    const card = el("article", "how-card how-card-benefit");
    const head = el("div", "how-sheet-head");
    head.append(el("p", "how-kicker", `${benefit.prompts} test prompts, ${benefit.candidates} images each`));
    head.append(el("h3", "how-title", "Does searching help in general?"));
    head.append(el("p", "how-text", "A little, and not everywhere. A separate detector and a separate CLIP model did the grading, so the search couldn't mark its own work."));
    const wrap = el("div", "how-table");
    const table = el("table");
    const headRow = el("tr");
    ["Correct", "One image", `Best of ${benefit.candidates}`].forEach((name) => {
      const th = el("th", "", name);
      th.scope = "col";
      headRow.append(th);
    });
    const thead = el("thead");
    thead.append(headRow);
    const tbody = el("tbody");
    const rows = benefit.rows;
    rows.forEach((row) => {
      const tr = el("tr", row.check === "all" ? "how-table-all" : "");
      const isClip = row.check === "clip_l14";
      const value = (v) => (isClip ? v.toFixed(2) : `${v.toFixed(1)}%`);
      tr.append(el("td", "", row.label), el("td", "", value(row.one)), el("td", "", value(row.best_of_8)));
      tbody.append(tr);
    });
    table.append(thead, tbody);
    wrap.append(table);
    const all = rows.find((r) => r.check === "all");
    const two = rows.find((r) => r.check === "two_object");
    const attr = rows.find((r) => r.check === "color_attr");
    const foot = el("p", "how-foot", `All checkable prompts: +${all.diff.toFixed(1)} points (95% CI ${all.ci95_lo.toFixed(1)} to ${all.ci95_hi.toFixed(1)}, n = ${all.n}). Two objects gained most, ${two.one.toFixed(1)}% to ${two.best_of_8.toFixed(1)}%. Colours on the right objects didn't improve (${attr.one.toFixed(1)}% to ${attr.best_of_8.toFixed(1)}%, within noise). Apple M4, original wording only. The iPhone app runs the same idea at 36 to 41 images in about 5 s.`);
    card.append(head, wrap, foot);
    return card;
  }

  const cards = [promptCard(), ...STOPS.map(attemptCard), sheetCard(), benefitCard()];
  const tabs = ["Prompt", ...STOPS.map((stop) => `Image ${stop.n}`), "All tries", "Test set"];
  stack.style.setProperty("--tab-slots", String(cards.length));
  cards.forEach((card, index) => stack.append(step(index, card, tabs[index])));
  note.textContent = "One recorded run of this website's pipeline on an Apple M4, picked because it shows each step. 3 of our 11 recorded runs of this prompt ended with every check passing. These are saved images, not live results.";

  box.hidden = false;
  // After unhiding, so each folder shape is measured at its real size.
  if (window.Folder) cards.forEach(window.Folder.attach);
  const link = document.getElementById("how-link");
  if (link) link.hidden = false;
  if (location.hash === "#how") box.scrollIntoView();
})();
