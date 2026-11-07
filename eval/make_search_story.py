#!/usr/bin/env python
"""Records real search runs of the website pipeline, keeping every candidate and its scores,
for the "How it finds the best image" section. Harmless prompt only. Images the safety gate
or the quality check rejects come back empty from the pipeline, so they can't be saved.

    PYTHONPATH=. python eval/make_search_story.py --seeds 42 --out /tmp/runs
    cp /tmp/runs/run_42/* web/search/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.modules["torch"] = None  # Same as the server: this path never needs it, and it saves memory.

import cv2  # noqa: E402

from kshana import classical, parser, planner, safety  # noqa: E402
from kshana.inputs import fit_tokens, sanitize  # noqa: E402
from kshana.pipeline import Pipeline  # noqa: E402

PROMPT = "a red apple to the left of a blue cup"
BUDGET = "best"
THREADS = 4
JPEG_QUALITY = 82


def record(pipeline: Pipeline, seed: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    clean = sanitize(PROMPT)
    fitted = fit_tokens(clean, pipeline.gen.truncate)
    texts = parser.rewordings(parser.parse(fitted.text)) + [fitted.text]
    attempts: list[dict] = []
    plan_box: dict = {}
    original = pipeline._attempt

    def recording(attempt_seed, emb, plan, parsed, text_emb):
        result = original(attempt_seed, emb, plan, parsed, text_emb)
        number = len(attempts) + 1
        entry = {
            "n": number,
            "seed": attempt_seed,
            "wording": texts[attempt_seed % len(texts)],
            "score": result.score if result.usable else None,
            "rejected": result.rejected,
            "new_best": False,
        }
        if result.image is not None:
            name = f"attempt_{number:02d}.jpg"
            shown = safety.stamp_label(classical.resize_lanczos(result.image, plan.deliver_res))
            cv2.imwrite(str(out / name), shown[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            entry["src"] = f"search/{name}"
            entry["checks"] = result.payload["checklist"]
            entry["classical"] = result.payload["classical"]
            entry["times"] = result.payload["times"]
        attempts.append(entry)
        return result

    def on_best(best: dict) -> None:
        attempts[best["attempt"] - 1]["new_best"] = True

    pipeline._attempt = recording
    try:
        outcome = pipeline.run(
            clean,
            BUDGET,
            planner.Device(),
            30.0,
            0,
            lambda plan, notes: plan_box.update(plan.as_dict()),
            on_best,
            lambda: False,
            first_seed=seed,
        )
    finally:
        pipeline._attempt = original
    story = {
        "prompt": PROMPT,
        "budget": BUDGET,
        "hardware": f"Apple M4, ONNX Runtime, {THREADS} CPU threads",
        "first_seed": seed,
        "plan": plan_box,
        "status": outcome.status,
        "timings": outcome.timings,
        "attempts": attempts,
    }
    (out / "story.json").write_text(json.dumps(story, indent=1))
    return story


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pipeline = Pipeline(threads=THREADS)
    pipeline.load()
    pipeline.benchmark()
    for seed in args.seeds:
        story = record(pipeline, seed, args.out / f"run_{seed}")
        print(f"\nseed {seed}: {story['status']}, {len(story['attempts'])} images, {story['timings'].get('compute_ms')} ms")
        for a in story["attempts"]:
            fails = [c["label"] for c in a.get("checks", []) if c["status"] == "fail"]
            print(f"  {a['n']:2d} score={a['score']} {'BEST' if a['new_best'] else '    '} {a['rejected'] or ''} fails={fails}")


if __name__ == "__main__":
    main()
