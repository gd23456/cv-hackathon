"use strict";

const API = (window.KSHANA_API || "").replace(/\/$/, "");
const POLL_MS = 1000;
const HEALTH_TIMEOUT_MS = 8000;
const WAKE_GIVE_UP_MS = 120000;
const MAX_POLL_FAILURES = 8; // the server cancels a job after 10 s without a poll
const PATCHES = 32;

const SAMPLES = [
  { src: "samples/apple-cup.jpg", prompt: "a red apple to the left of a blue cup on a wooden table" },
  { src: "samples/two-dogs.jpg", prompt: "two dogs on a green sofa" },
  { src: "samples/cabin.jpg", prompt: "a small cabin beside a lake at sunset, mountains in the background" },
  { src: "samples/puppy.jpg", prompt: "a golden retriever puppy sitting on green grass, sunny day" },
];

const BUDGET_HINTS = {
  auto: "Chosen from your device and your prompt.",
  fast: "Up to 4 seconds. Fewer attempts.",
  balanced: "Up to 10 seconds.",
  best: "Up to 24 seconds. More attempts to choose from.",
};

const $ = (id) => document.getElementById(id);
const el = {
  shell: document.querySelector(".shell"), report: $("report"),
  server: $("server"), serverText: $("server-text"),
  form: $("form"), prompt: $("prompt"), promptError: $("prompt-error"),
  go: $("go"), budgetHint: $("budget-hint"),
  frame: $("frame"), plate: $("plate"), slate: $("slate"),
  strip: $("strip"), caption: $("caption"), live: $("live"),
  samples: $("samples"), samplesGrid: $("samples-grid"),
  tier: $("tier"), tierName: $("tier-name"), tierReason: $("tier-reason"), notes: $("notes"),
  summary: $("check-summary"), checks: $("checks"),
  times: $("times"), timesEmpty: $("times-empty"),
  cleanup: $("cleanup"), cleanupList: $("cleanup-list"),
};

const state = {
  server: "checking",
  downSince: null,
  jobId: null,
  version: null,
  job: null,
  active: false,
  startedAt: null,
  budgetS: null,
  notches: [],
  pollFailures: 0,
  waitTimer: null,
  stripTimer: null,
  prompt: "",
};

// ---- small helpers ---------------------------------------------------

function make(tag, props = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  node.append(...kids);
  return node;
}

function formatMs(ms) {
  if (ms == null) return "–";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

function reduced() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

async function fetchJson(path, options = {}, timeoutMs = 10000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(API + path, { ...options, signal: ctrl.signal, cache: "no-store" });
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON error page */ }
    return { status: res.status, body };
  } finally {
    clearTimeout(timer);
  }
}

// ---- server health ---------------------------------------------------

function setServer(mode, text) {
  state.server = mode;
  el.server.dataset.state = mode;
  el.serverText.textContent = text;
  const offline = mode === "down";
  el.shell.toggleAttribute("data-offline", offline && !state.job);
  el.samples.hidden = !(offline && !state.job);
  if (offline && !el.samplesGrid.childElementCount) renderSamples();
  renderButton();
}

async function checkHealth() {
  try {
    const { status, body } = await fetchJson("/health", {}, HEALTH_TIMEOUT_MS);
    if (status === 200 && body && body.ready) {
      state.downSince = null;
      setServer("ready", "Server ready");
      return setTimeout(checkHealth, 30000);
    }
    if (status === 200 && body && body.status === "warming") {
      setServer("waking", "Server warming up, about a minute");
      return setTimeout(checkHealth, 2000);
    }
    throw new Error(`health ${status}`);
  } catch {
    state.downSince = state.downSince || Date.now();
    if (Date.now() - state.downSince < WAKE_GIVE_UP_MS) {
      setServer("waking", "Waking the server, this can take a minute");
      return setTimeout(checkHealth, 4000);
    }
    setServer("down", "Server offline, showing samples");
    return setTimeout(checkHealth, 15000);
  }
}

function renderSamples() {
  el.samplesGrid.replaceChildren(
    ...SAMPLES.map((s) =>
      make("figure", {},
        make("img", { src: s.src, alt: `Sample image, made earlier: ${s.prompt}`, loading: "lazy", width: "512", height: "512" }),
        make("figcaption", {}, make("strong", { text: "Sample, made earlier" }), s.prompt))
    )
  );
}

// ---- rendering -------------------------------------------------------

function buildStrip() {
  el.strip.replaceChildren(...Array.from({ length: PATCHES }, () => make("i")));
}

function paintStrip(fraction) {
  const f = Math.max(0, Math.min(1, fraction));
  const on = Math.round(f * PATCHES);
  [...el.strip.querySelectorAll("i")].forEach((p, i) => p.classList.toggle("on", i < on));
  el.strip.setAttribute("aria-valuenow", String(Math.round(f * 100)));
}

function addNotch(fraction) {
  const f = Math.max(0, Math.min(1, fraction));
  el.strip.append(make("b", { style: `left:${(f * 100).toFixed(1)}%` }));
}

function stopStrip() {
  if (state.stripTimer) clearInterval(state.stripTimer);
  state.stripTimer = null;
}

function startStrip() {
  stopStrip();
  state.stripTimer = setInterval(() => {
    if (!state.startedAt || !state.budgetS) return;
    paintStrip((performance.now() - state.startedAt) / (state.budgetS * 1000));
    updateElapsed();
  }, 200);
}

function updateElapsed() {
  const node = el.slate.querySelector(".slate-num");
  if (node && state.startedAt) node.textContent = `${((performance.now() - state.startedAt) / 1000).toFixed(1)} s`;
}

function slate(...nodes) {
  el.slate.replaceChildren(...nodes);
}

function registerMark() {
  const svg = (cls) => {
    const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    s.setAttribute("class", cls);
    s.setAttribute("aria-hidden", "true");
    const u = document.createElementNS("http://www.w3.org/2000/svg", "use");
    u.setAttribute("href", "#i-reg");
    s.append(u);
    return s;
  };
  return make("div", { class: "reg" }, svg("r-c"), svg("r-m"), svg("r-y"));
}

function showImage(best, alt) {
  const src = `data:${best.mime || "image/jpeg"};base64,${best.image_b64}`;
  const plain = make("img", { src, alt, width: String(best.width), height: String(best.height) });
  if (reduced()) {
    el.plate.replaceChildren(plain);
    return;
  }
  // The new image comes in as three separated plates and settles into register.
  const plates = ["c", "m", "y"].map((c) => make("img", { src, alt: "", class: `sep sep-${c}` }));
  let settled = false;
  const finish = () => {
    if (settled) return;
    settled = true;
    el.plate.classList.remove("sep-bg");
    el.plate.replaceChildren(plain);
  };
  plates[0].addEventListener("animationend", finish, { once: true });
  setTimeout(finish, 900);
  el.plate.classList.add("sep-bg");
  el.plate.replaceChildren(...plates);
}

function clearResultPanels() {
  el.tier.hidden = true;
  el.notes.hidden = true;
  el.notes.replaceChildren();
  el.summary.textContent = "Checks appear with the first image.";
  el.checks.replaceChildren();
  el.times.replaceChildren();
  el.timesEmpty.hidden = false;
  el.cleanup.hidden = true;
  el.cleanupList.replaceChildren();
  el.caption.textContent = "";
}

const CHECK_MARK = { pass: "#i-pass", fail: "#i-fail", not_verifiable: "#i-none", info: "#i-info" };
const CHECK_WORD = { pass: "passed", fail: "failed", not_verifiable: "can't be checked", info: "" };

function checkRow(item) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("aria-hidden", "true");
  const u = document.createElementNS("http://www.w3.org/2000/svg", "use");
  u.setAttribute("href", CHECK_MARK[item.status] || "#i-none");
  svg.append(u);
  const word = CHECK_WORD[item.status] ?? "";
  const label = make("div", { class: "check-label" }, item.label);
  if (item.kind === "alignment" && item.value != null) {
    label.append(make("span", { class: "check-state", text: ` ${item.value.toFixed(2)}` }));
  } else if (word) {
    label.append(make("span", { class: "check-state", text: ` ${word}` }));
  }
  return make("li", { class: "check" }, svg, label, make("p", { class: "check-detail", text: item.detail || "" }));
}

function renderChecklist(items) {
  if (!items || !items.length) return;
  const n = (s) => items.filter((i) => i.status === s).length;
  const parts = [];
  if (n("pass")) parts.push(`${n("pass")} passed`);
  if (n("fail")) parts.push(`${n("fail")} failed`);
  if (n("not_verifiable")) parts.push(`${n("not_verifiable")} can't be checked`);
  el.summary.textContent = parts.join(", ") || "Only the overall match could be measured.";
  el.checks.replaceChildren(...items.map(checkRow));
}

function renderTimes(t) {
  if (!t) return;
  const rows = [];
  if (t.queue_ms != null) rows.push(["Waited in queue", formatMs(t.queue_ms)]);
  if (t.first_image_ms != null) rows.push(["First image, after queue", formatMs(t.first_image_ms)]);
  if (t.total_ms != null) rows.push(["Total", formatMs(t.total_ms)]);
  if (t.attempts != null) rows.push(["Attempts", String(t.attempts)]);
  if (t.last_attempt) {
    rows.push(["Generate, best attempt", formatMs(t.last_attempt.generate_ms)]);
    rows.push(["Clean-up, best attempt", formatMs(t.last_attempt.classical_ms)]);
    rows.push(["Checks, best attempt", formatMs(t.last_attempt.verify_ms)]);
  }
  if (!rows.length) return;
  el.timesEmpty.hidden = true;
  el.times.replaceChildren(...rows.flatMap(([k, v]) => [make("dt", { text: k }), make("dd", { text: v })]));
}

function renderCleanup(ops) {
  if (!ops || !ops.length) return;
  el.cleanup.hidden = false;
  el.cleanupList.replaceChildren(
    ...ops.map((o) => make("li", {}, `${o.op.replace("_", " ")}: ${o.applied ? "ran" : "did not run"} `, make("span", { text: `(${o.note})` })))
  );
}

function renderTier(job) {
  if (!job.tier) return;
  el.tier.hidden = false;
  el.tierName.textContent = job.tier;
  el.tierReason.textContent = (job.reason || "").replace(/^\w+:\s*/, "");
}

function renderNotes(notes) {
  if (!notes || !notes.length) return;
  el.notes.hidden = false;
  el.notes.replaceChildren(...notes.map((n) => make("li", { text: n })));
}

function renderButton() {
  const blocked = state.server === "down";
  if (state.active) {
    el.go.disabled = true;
    el.go.textContent = "Generating";
  } else if (state.waitTimer) {
    el.go.disabled = true;
  } else {
    el.go.disabled = blocked;
    el.go.textContent = blocked ? "Server offline" : state.job ? "Generate again" : "Generate";
  }
}

function setPhase(phase) {
  el.frame.dataset.phase = phase;
}

function announce(text) {
  if (el.live.textContent !== text) el.live.textContent = text;
}

function finishJob() {
  state.active = false;
  stopStrip();
  renderButton();
}

function showFailure(phase, big, text, retry) {
  finishJob();
  setPhase(phase);
  el.plate.replaceChildren();
  el.report.hidden = true; // no image, so nothing to report
  const nodes = [make("p", { class: "slate-big", text: big })];
  if (text) nodes.push(make("p", { class: "slate-text", text }));
  announce(text ? `${big}. ${text}` : big);
  if (retry) {
    const btn = make("button", { class: "slate-btn", type: "button", text: "Try again" });
    btn.addEventListener("click", submit);
    nodes.push(btn);
  }
  slate(...nodes);
}

function renderJob(job) {
  state.job = job;
  renderTier(job);
  renderNotes(job.notes);
  renderChecklist(job.checklist);
  renderTimes(job.timings);
  renderCleanup(job.classical);

  const plan = job.timings && job.timings.plan;
  if (plan) state.budgetS = plan.budget_s;
  if (job.status === "running" && !state.startedAt) {
    state.startedAt = performance.now();
    startStrip();
  }

  if (job.best && job.best.image_b64 && job.best.version !== state.version) {
    state.version = job.best.version;
    slate();
    announce("New image ready.");
    showImage(job.best, `Generated image for the prompt: ${state.prompt}`);
    if (state.startedAt && state.budgetS) addNotch((performance.now() - state.startedAt) / (state.budgetS * 1000));
  }
  if (job.best) {
    const verb = job.status === "done" ? "Best of" : "Best so far of";
    const tried = job.timings && job.timings.attempts ? job.timings.attempts : job.best.attempt;
    el.caption.textContent = `${verb} ${tried} ${tried === 1 ? "attempt" : "attempts"} · ${job.best.width} px · AI-generated`;
  }

  switch (job.status) {
    case "queued": {
      setPhase("queued");
      const ahead = Math.max((job.queue_position || 1) - 1, 0);
      announce(ahead === 0 ? "You're next." : `${ahead} ahead of you.`);
      slate(
        registerMark(),
        make("p", { class: "slate-big", text: ahead === 0 ? "You're next" : `${ahead} ahead of you` }),
        make("p", { class: "slate-text", text: "One image is made at a time on this server." })
      );
      break;
    }
    case "running":
      if (!job.best) {
        if (el.frame.dataset.phase !== "running") {
          setPhase("running");
          announce("Making the first image.");
          slate(
            registerMark(),
            make("p", { class: "slate-big slate-num", text: "0.0 s" }),
            make("p", { class: "slate-text", text: "Making the first image." })
          );
        }
      } else {
        setPhase("improving");
      }
      break;
    case "done":
      finishJob();
      setPhase("done");
      paintStrip(1);
      slate();
      announce(el.caption.textContent ? `Finished. ${el.caption.textContent}` : "Finished.");
      break;
    case "refused":
      showFailure("refused", "Can't make this", job.error);
      el.caption.textContent = "";
      paintStrip(0);
      break;
    case "timeout":
      if (job.best) { finishJob(); setPhase("done"); paintStrip(1); }
      else showFailure("error", "Ran out of time", job.error, true);
      break;
    case "cancelled":
      if (job.best) { finishJob(); setPhase("done"); }
      else showFailure("error", "Job stopped", "This page stopped checking in, perhaps because it was in the background.", true);
      break;
    default:
      showFailure("error", "Something went wrong", job.error || "The server could not finish this image.", true);
  }
}

// ---- polling ---------------------------------------------------------

async function poll() {
  if (!state.active || !state.jobId) return;
  const since = state.version != null ? `?since=${state.version}` : "";
  try {
    const { status, body } = await fetchJson(`/api/jobs/${state.jobId}${since}`, {}, 5000);
    if (status === 200 && body) {
      state.pollFailures = 0;
      renderJob(body);
    } else if (status === 404) {
      showFailure("error", "The server restarted", "This job was lost. Please try again.", true);
      return;
    } else {
      throw new Error(`poll ${status}`);
    }
  } catch {
    state.pollFailures += 1;
    if (state.pollFailures >= MAX_POLL_FAILURES) {
      showFailure("error", "Lost contact with the server", "Check your connection and try again.", true);
      return;
    }
  }
  if (state.active) setTimeout(poll, POLL_MS);
}

// ---- submit ----------------------------------------------------------

function deviceHints() {
  const c = navigator.connection || {};
  return {
    memory_gb: navigator.deviceMemory ?? null,
    cores: navigator.hardwareConcurrency ?? null,
    save_data: Boolean(c.saveData),
  };
}

function showFieldError(msg) {
  el.promptError.textContent = msg;
  el.promptError.hidden = false;
  el.prompt.setAttribute("aria-invalid", "true");
  el.prompt.focus();
}

function clearFieldError() {
  el.promptError.hidden = true;
  el.prompt.removeAttribute("aria-invalid");
}

function countdown(big, text, seconds) {
  let left = Math.max(1, Math.round(seconds));
  showFailure("error", big, text);
  const tick = () => {
    if (left <= 0) {
      clearInterval(state.waitTimer);
      state.waitTimer = null;
      renderButton();
      return;
    }
    el.go.textContent = `Try again in ${left} s`;
    el.go.disabled = true;
    left -= 1;
  };
  if (state.waitTimer) clearInterval(state.waitTimer);
  state.waitTimer = setInterval(tick, 1000);
  tick();
}

function revealResult() {
  // On a phone the frame is below the form. Wait a frame for the keyboard to start closing.
  if (window.matchMedia("(max-width: 719px)").matches) {
    requestAnimationFrame(() => el.frame.scrollIntoView({ behavior: reduced() ? "auto" : "smooth", block: "start" }));
  }
}

async function submit() {
  const text = el.prompt.value.trim();
  clearFieldError();
  if (!text) {
    showFieldError("Type what you want to see, for example: a red bicycle by a lake.");
    return;
  }
  if (state.active || state.waitTimer) return;

  state.prompt = text;
  state.jobId = null;
  state.version = null;
  state.job = null;
  state.startedAt = null;
  state.budgetS = null;
  state.pollFailures = 0;
  state.active = true;
  el.shell.removeAttribute("data-offline");
  el.samples.hidden = true;
  el.report.hidden = false;
  el.plate.replaceChildren();
  clearResultPanels();
  buildStrip();
  paintStrip(0);
  setPhase("submitting");
  slate(registerMark(), make("p", { class: "slate-big", text: "Sending" }));
  renderButton();
  el.prompt.blur();
  revealResult();

  const budget = el.form.querySelector('input[name="budget"]:checked').value;
  let res;
  try {
    res = await fetchJson(
      "/api/generate",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text, budget, device: deviceHints() }),
      },
      20000
    );
  } catch {
    showFailure("error", "Can't reach the server", "It may be waking up. Please try again in a moment.", true);
    return;
  }

  const { status, body } = res;
  const code = body && body.error && body.error.code;
  const message = body && body.error && body.error.message;
  if (status === 200 && body && body.job_id) {
    state.jobId = body.job_id;
    poll();
  } else if (status === 422) {
    finishJob();
    setPhase("idle");
    slate(make("p", { class: "slate-big", text: "Your image goes here" }));
    showFieldError(message || "That prompt can't be used. Please change it and try again.");
  } else if (status === 429) {
    countdown("That's the limit for now", message, body && body.retry_after ? body.retry_after : 30);
  } else if (code === "busy") {
    countdown("Server is busy", "It makes one image at a time and the queue is full.", body.retry_after || 15);
  } else if (code === "warming") {
    countdown("Server is warming up", "It will be ready shortly.", body.retry_after || 5);
  } else {
    showFailure("error", "Something went wrong", message || `The server answered with an error (${status}).`, true);
  }
}

// ---- events ----------------------------------------------------------

el.form.addEventListener("submit", (e) => {
  e.preventDefault();
  submit();
});

el.prompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    submit();
  }
});

el.prompt.addEventListener("input", clearFieldError);

document.querySelectorAll(".example").forEach((b) =>
  b.addEventListener("click", () => {
    el.prompt.value = b.dataset.prompt;
    clearFieldError();
  })
);

el.form.querySelectorAll('input[name="budget"]').forEach((r) =>
  r.addEventListener("change", () => {
    el.budgetHint.textContent = BUDGET_HINTS[r.value];
  })
);

// Bring the poll back immediately when a phone returns to this tab.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && state.active && state.jobId) poll();
});

buildStrip();
slate(
  make("p", { class: "slate-big", text: "Your image goes here" }),
  make("p", { class: "slate-text", text: "It appears as soon as the first attempt is ready, then gets replaced whenever a later attempt scores higher." })
);
checkHealth();
