"""Backup demo for Streamlit Community Cloud: the same kshana pipeline, no FastAPI.

Local run: KSHANA_MODELS=server/models streamlit run streamlit_app.py
On a Mac, MallocMediumZone=0 MallocLargeCache=0 stops macOS holding on to freed pages, as Linux does not.
"""
import base64
import os
import threading
import time
from collections import deque
from pathlib import Path

# Before kshana loads: 2 shared cores, and models must not share the 2.7 GB limit.
os.environ.setdefault("KSHANA_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", os.environ["KSHANA_THREADS"])
os.environ["KSHANA_LOW_MEMORY"] = "1"
# Outside the repo, so 2 GB of downloads can never be committed by accident.
os.environ.setdefault("KSHANA_MODELS", str(Path.home() / ".cache" / "kshana" / "models"))

import streamlit as st  # noqa: E402

from kshana import weights  # noqa: E402
from kshana.inputs import InputError, sanitize  # noqa: E402
from kshana.pipeline import Pipeline  # noqa: E402
from kshana.planner import Device  # noqa: E402
from kshana.safety import check_prompt  # noqa: E402

JOB_LIMIT_S = 30.0
QUEUE_WAIT_S = 45.0
RATE_LIMIT = 5
MARK = {"pass": "✅", "fail": "❌", "not_verifiable": "➖", "info": "📈"}
STATE_WORD = {"pass": "passed", "fail": "failed", "not_verifiable": "can't be checked", "info": ""}


class Engine:
    """Boots once in the background and is shared by every session. Makes one image at a time."""

    def __init__(self):
        self.status = "Starting"
        self.error = None
        self.pipeline = None
        self.lock = threading.Lock()
        threading.Thread(target=self._boot, daemon=True).start()

    def _boot(self):
        try:
            models = Path(os.environ["KSHANA_MODELS"])
            if weights.missing(models):
                weights.ensure(models, self._progress)
            self.status = "Loading the models"
            pipeline = Pipeline(models)
            pipeline.load()
            self.status = "Measuring this server's speed, to choose a resolution"
            costs = pipeline.benchmark()
            print("kshana ready, measured costs:", costs.as_dict(), flush=True)  # for the host's log
            self.pipeline = pipeline
        except Exception as e:  # shown on the page
            self.error = str(e)

    def _progress(self, name, total):
        self.status = f"Downloading the model weights, about 2 GB: {total / 1e9:.2f} GB so far"


@st.cache_resource(show_spinner=False)
def engine() -> Engine:
    return Engine()


def over_rate_limit() -> bool:
    hits = st.session_state.setdefault("hits", deque())
    now = time.monotonic()
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        return True
    hits.append(now)
    return False


def show_checks(slot, items):
    if not items:
        return
    lines = []
    for item in items:
        value = f" {item['value']:.2f}" if item["kind"] == "alignment" and "value" in item else ""
        word = f" {STATE_WORD[item['status']]}" if STATE_WORD[item["status"]] else ""
        lines.append(f"{MARK[item['status']]} **{item['label']}**{value}{word}  \n{item['detail']}")
    slot.markdown("\n\n".join(lines))


def show_times(slot, t, best):
    rows = [
        ("First image", t.get("first_image_ms")),
        ("Total", t.get("compute_ms")),
        ("Generate, best attempt", best["attempt_times"]["generate_ms"]),
        ("Clean-up, best attempt", best["attempt_times"]["classical_ms"]),
        ("Checks, best attempt", best["attempt_times"]["verify_ms"]),
    ]
    table = "| Step | Time |\n|---|---:|\n" + "\n".join(f"| {k} | {v / 1000:.1f} s |" for k, v in rows if v is not None)
    n = t.get("attempts", 1)
    slot.markdown(f"{table}\n\n{n} attempt{'' if n == 1 else 's'} · measured live on this server")


def generate(eng, prompt, budget):
    try:
        clean = sanitize(prompt)
    except InputError as e:
        st.warning(e.message)
        return
    if over_rate_limit():
        st.warning(f"That's {RATE_LIMIT} images this minute, the limit per person. Please wait a moment.")
        return
    category = check_prompt(clean.text)
    if category:
        st.error(f"This prompt can't be generated because it asks for {category}.")
        return

    image_slot, status_slot = st.empty(), st.empty()
    reason_slot = st.empty()
    notes_slot = st.empty()
    st.subheader("What was checked")
    checks_slot = st.empty()
    st.subheader("Timings")
    times_slot = st.empty()

    status_slot.info("Waiting for the image ahead of yours. This server makes one at a time.")
    if not eng.lock.acquire(timeout=QUEUE_WAIT_S):
        status_slot.warning("The server is busy with other people's images. Please try again soon.")
        return
    try:
        status_slot.info("Making the first image")
        started = time.monotonic()
        latest = {}

        def on_plan(plan, notes):
            reason_slot.markdown(f"**{plan.tier}.** {plan.reason.split(': ', 1)[-1]}")
            if notes:
                notes_slot.warning(" ".join(notes))

        def on_best(best):
            latest.update(best)
            image_slot.image(base64.b64decode(best["image_b64"]), caption=f"Best so far: attempt {best['attempt']} · {best['width']} px · AI-generated")
            status_slot.info(f"Trying for a better one. {time.monotonic() - started:.1f} s so far")
            show_checks(checks_slot, best["checklist"])

        outcome = eng.pipeline.run(clean, budget, Device(), JOB_LIMIT_S, 0, on_plan, on_best, lambda: False)
    finally:
        eng.lock.release()

    if outcome.status == "refused":
        status_slot.error(outcome.error)
    elif outcome.status != "done":
        status_slot.error(outcome.error or "Something went wrong. Please try again.")
    else:
        status_slot.success(f"Done in {outcome.timings['compute_ms'] / 1000:.1f} s")
        show_times(times_slot, outcome.timings, latest)


st.set_page_config(page_title="Kshana", layout="centered")
st.title("Kshana")
st.write("Type a prompt. The image improves until time runs out, and the checks say what it got right.")

eng = engine()
if eng.error:
    st.error(f"The server could not start: {eng.error}")
    st.stop()
if eng.pipeline is None:
    st.info(f"{eng.status}. The first visit after a restart takes a few minutes.")
    time.sleep(2)
    st.rerun()

with st.form("make"):
    prompt = st.text_area("Prompt", placeholder="a red apple to the left of a blue cup", max_chars=1000, height=90)
    label = st.radio("Time to spend", ["Auto", "Fast", "Balanced", "Best"], horizontal=True, help="Fast is up to 4 s, Balanced up to 10 s, Best up to 24 s.")
    submitted = st.form_submit_button("Generate", type="primary", use_container_width=True)

if submitted:
    generate(eng, prompt, label.lower())
