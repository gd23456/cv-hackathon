#!/usr/bin/env python
"""Measures prompt rewording against the original wording, one image per prompt, same seeds.
Scored by the independent GenEval-style scorer. Generates to disk, scores, then deletes images."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PIL import Image

from eval.generate_coreml import CoreMLGenerator
from eval.geneval_score import GEN, read_split, score_config, split_name, tag_means
from kshana import parser


def run(generator, split: str, config: str, reword: bool) -> dict:
    out = GEN / config / split_name(split)
    out.mkdir(parents=True, exist_ok=True)
    changed = 0
    for row in read_split(split):
        path = out / f"{row['id']}.png"
        if path.exists():
            continue
        prompt = row["prompt"]
        if reword:
            lines = parser.rewordings(parser.parse(prompt))
            if lines:
                prompt = lines[0]
                changed += 1
        image, _ = generator.generate(prompt, row["seed"])
        Image.fromarray(image).save(path)
    means = tag_means(score_config(config, split))
    shutil.rmtree(out)
    print(config, split, "reworded", changed, {k: (round(v[0], 3), v[1]) for k, v in means.items()}, flush=True)
    return means


def main() -> None:
    generator = CoreMLGenerator("sdxs-dreamshaper", "A")
    rows = []
    for split in ("complex_dev", "complex_test"):
        for config, reword in (("S1-A-original", False), ("S1-A-reworded", True)):
            for tag, (accuracy, n) in run(generator, split, config, reword).items():
                rows.append({"config": config, "split": split, "tag": tag, "accuracy": round(accuracy, 4), "n": n})
    with Path("results/complex_rewrite.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
