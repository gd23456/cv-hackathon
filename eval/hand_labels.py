#!/usr/bin/env python
"""Export about 100 images for hand labelling, and score the labels against the detector.

    python eval/hand_labels.py export --config S1-A --split complex_dev --n 100
    python eval/hand_labels.py agreement --config S1-A --split complex_dev

export writes results/hand_labels/<config>_<split>.csv and copies the images to data/hand_labels/<config>_<split>/.
Fill the hand_* columns with 1, 0, or leave blank when a check does not apply. The scorer's output is not in the
sheet, so labelling stays blind.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, GEN, RESULTS, guard_test_split, read_split, split_name  # noqa: E402

SEED = 7
CHECKS = ("presence", "count", "position", "color")


def sheet_path(config: str, split: str) -> Path:
    return RESULTS / "hand_labels" / f"{config}_{split}.csv"


def export(config: str, split: str, n: int) -> None:
    rows = [r for r in read_split(split) if r["include"] and r.get("verifiable", True)]
    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(len(rows), size=min(n, len(rows)), replace=False).tolist())
    name = split_name(split)
    img_out = DATA / "hand_labels" / f"{config}_{name}"
    img_out.mkdir(parents=True, exist_ok=True)
    out = sheet_path(config, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image", "tag", "prompt", *[f"hand_{c}" for c in CHECKS], "hand_notes"])
        for i in chosen:
            r = rows[i]
            src = GEN / config / name / f"{r['id']}.png"
            shutil.copy2(src, img_out / src.name)
            w.writerow([r["id"], str((img_out / src.name).relative_to(DATA.parent)), r["tag"], r["prompt"], "", "", "", "", ""])
    print(out, len(chosen))


def agreement(config: str, split: str) -> None:
    name = split_name(split)
    with open(sheet_path(config, name), newline="") as f:
        hand = {r["id"]: r for r in csv.DictReader(f)}
    with open(DATA / "scores" / config / f"{name}.csv", newline="") as f:
        auto = {r["id"]: r for r in csv.DictReader(f)}
    for check in CHECKS:
        pairs = [(hand[i][f"hand_{check}"], auto[i].get(check, "")) for i in hand if hand[i][f"hand_{check}"] in ("0", "1") and auto.get(i, {}).get(check) in ("True", "False")]
        if not pairs:
            print(check, "no labels")
            continue
        h = np.array([p[0] == "1" for p in pairs])
        a = np.array([p[1] == "True" for p in pairs])
        print(f"{check}: n={len(pairs)} agreement={float((h == a).mean()):.3f} detector_pass={float(a.mean()):.3f} hand_pass={float(h.mean()):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "agreement"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()
    guard_test_split(args.split, args.final_test)
    if args.mode == "export":
        export(args.config, args.split, args.n)
    else:
        agreement(args.config, args.split)


if __name__ == "__main__":
    main()
