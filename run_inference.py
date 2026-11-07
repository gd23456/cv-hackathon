#!/usr/bin/env python
"""Generate images from a list of prompts, with the same pipeline and safety checks as the website.

    python run_inference.py --prompts samples/prompts.txt --output_dir results/samples
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from kshana import planner, safety, weights
from kshana.inputs import InputError, sanitize
from kshana.pipeline import Pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", type=Path, default=Path("samples/prompts.txt"), help="text file with one prompt per line")
    ap.add_argument("--output_dir", type=Path, default=Path("results/samples"))
    ap.add_argument("--budget", choices=["auto", "fast", "balanced", "best"], default="balanced")
    ap.add_argument("--models", type=Path, default=Path("server/models"))
    args = ap.parse_args()

    if weights.missing(args.models):
        print(f"Downloading weights to {args.models}...")
        try:
            weights.ensure(args.models)
        except OSError as error:
            sys.exit(f"Couldn't download the model files ({error}). Build them instead: see Quickstart in README.md.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(args.models)
    pipeline.load()
    pipeline.benchmark()

    prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    report = []
    for index, prompt in enumerate(prompts):
        best: dict = {}
        try:
            clean = sanitize(prompt)
        except InputError as error:
            report.append({"prompt": prompt, "status": "refused", "reason": str(error)})
            print(f"[{index}] refused: {error}")
            continue
        category = safety.check_prompt(clean.text)
        if category:
            report.append({"prompt": prompt, "status": "refused", "reason": category})
            print(f"[{index}] refused: {category}")
            continue
        outcome = pipeline.run(
            clean, args.budget, planner.Device(), 30.0, 0,
            on_plan=lambda plan, notes: None, on_best=best.update, should_stop=lambda: False,
        )
        entry = {"prompt": prompt, "status": outcome.status, "error": outcome.error, "timings": outcome.timings}
        if outcome.status == "done" and best:
            path = args.output_dir / f"{index:02d}.jpg"
            path.write_bytes(base64.b64decode(best["image_b64"]))
            entry.update(image=str(path), checklist=best["checklist"], seed=best["seed"], score=best["score"])
        report.append(entry)
        print(f"[{index}] {outcome.status}: {prompt}")

    (args.output_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"Wrote {args.output_dir}/report.json")
    sys.exit(0 if all(r["status"] in ("done", "refused") for r in report) else 1)


if __name__ == "__main__":
    main()
