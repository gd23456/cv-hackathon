#!/usr/bin/env python
"""Best of 8 against one image on the held-out complex prompts, from data/candidates/S1-A/complex_test.npz.
The search picks with the runtime score (YOLOX + ViT-B/32). DETR-R101 and CLIP L/14 do the grading, so they're independent.

Writes results/search_benefit.csv and web/search/benefit.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
K = 8
RESAMPLES = 5000
TAGS = (
    ("single_object", "Single objects"),
    ("colors", "Single colours"),
    ("two_object", "Two objects"),
    ("counting", "Counts"),
    ("spatial", "Spatial"),
    ("color_attr", "Colours on the right objects"),
    ("position", "Positions"),
)


def paired(after: np.ndarray, before: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    diff = after - before
    boots = rng.choice(diff, (RESAMPLES, len(diff))).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def row(name: str, label: str, one: np.ndarray, best: np.ndarray, scale: float, rng: np.random.Generator) -> dict:
    diff, lo, hi = paired(best.astype(float), one.astype(float), rng)
    return {
        "check": name,
        "label": label,
        "n": len(one),
        "one": round(float(one.mean()) * scale, 2),
        "best_of_8": round(float(best.mean()) * scale, 2),
        "diff": round(diff * scale, 2),
        "ci95_lo": round(lo * scale, 2),
        "ci95_hi": round(hi * scale, 2),
    }


def main() -> None:
    with np.load(ROOT / "data/candidates/S1-A/complex_test.npz") as f:
        store = {k: f[k] for k in f.files}
    with open(ROOT / "eval/splits/complex_test.jsonl") as f:
        tag_by_id = {r["id"]: r["tag"] for r in map(json.loads, f)}
    tags = np.array([tag_by_id[i] for i in store["ids"].tolist()])

    rows_index = np.arange(len(tags))
    chosen = store["sel_both"][:, :K].argmax(axis=1)
    correct_one, correct_best = store["det_correct"][:, 0], store["det_correct"][rows_index, chosen]
    clip_one = np.maximum(store["clip_l14"][:, 0], 0)
    clip_best = np.maximum(store["clip_l14"][rows_index, chosen], 0)

    rng = np.random.default_rng(0)
    rows = []
    for tag, label in TAGS:
        usable = (tags == tag) & (correct_one >= 0) & (correct_best >= 0)
        rows.append(row(tag, label, correct_one[usable], correct_best[usable], 100, rng))
    usable = (correct_one >= 0) & (correct_best >= 0)
    rows.append(row("all", "All checkable prompts", correct_one[usable], correct_best[usable], 100, rng))
    rows.append(row("clip_l14", "CLIP L/14 score", clip_one, clip_best, 1, rng))

    with open(ROOT / "results/search_benefit.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    out = ROOT / "web/search/benefit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"prompts": len(tags), "candidates": K, "rows": rows}, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
