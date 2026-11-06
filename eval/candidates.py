#!/usr/bin/env python
"""Generate K candidate images per prompt and store every per-candidate metric, so best-of-N for all N and selectors
is computed afterwards without regenerating. Streaming and resumable.

    python eval/candidates.py --config S1-A --split complex_dev --k 8
    python eval/candidates.py --config B1 --split complex_dev --k 8

Candidate j uses seed + 100000*j, so candidate 0 is the same (prompt, seed) as the single-image run.
Output: data/candidates/<config>/<split>.npz, arrays shaped (prompts, k):
sel_clip, sel_detector, sel_both, clip_l14, and for complex prompts det_correct/presence/count/position/color (-1 = n/a).
Coverage with --inception adds inception (prompts, k, 2048).
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, wait_for_disk, guard_test_split, read_split, split_name  # noqa: E402

CHUNK_PROMPTS = 50
SEED_STRIDE = 100000
CHECKS = ("correct", "presence", "count", "position", "color")


def seed_for(base: int, j: int) -> int:
    return base + SEED_STRIDE * j


def tri(value) -> int:
    return -1 if value is None else int(bool(value))


def best_index(sel: np.ndarray, n: int) -> np.ndarray:
    """Index of the selected candidate among the first n, per prompt. Ties go to the earlier candidate."""
    return sel[:, :n].argmax(axis=1)


def run(config: str, split: str, k: int, inception: bool) -> None:
    from clip_models import load_clip, score_batch
    from generators import make_generator
    from geneval_score import Scorer, is_verifiable
    from selector import RuntimeVerifierSelector

    rows = read_split(split)
    name = split_name(split)
    out_dir = DATA / "candidates" / config
    parts = out_dir / f"{name}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    gen = make_generator(config)
    selector = RuntimeVerifierSelector()
    is_complex = "include" in rows[0]
    scorer = Scorer() if is_complex else None
    clip = scorer.clip if scorer else None
    processor = scorer.clip_processor if scorer else None
    if clip is None:
        clip, processor = load_clip("l14")
    inc = None
    if inception:
        from inception import InceptionFeatures

        inc = InceptionFeatures()
    t0 = time.time()
    for start in range(0, len(rows), CHUNK_PROMPTS):
        chunk = rows[start : start + CHUNK_PROMPTS]
        part = parts / f"{start:06d}.npz"
        if part.exists():
            continue
        wait_for_disk()
        store = {key: np.zeros((len(chunk), k), np.float32) for key in ("sel_clip", "sel_detector", "sel_both", "clip_l14")}
        if is_complex:
            store.update({f"det_{c}": np.full((len(chunk), k), -1, np.int8) for c in CHECKS})
        store["ms"] = np.zeros((len(chunk), k), np.float32)
        features = []
        with tempfile.TemporaryDirectory(dir=DATA) as tmp:
            for i, r in enumerate(chunk):
                images = []
                for j in range(k):
                    img, ms = gen.generate(r["prompt"], seed_for(r["seed"], j))
                    images.append(img)
                    store["ms"][i, j] = ms["total_ms"]
                    for s, v in selector.scores(img, r["prompt"]).items():
                        store[f"sel_{s}"][i, j] = v
                    if is_complex and is_verifiable(r):
                        path = Path(tmp) / f"{j}.png"
                        Image.fromarray(img).save(path)
                        res = scorer.score_image(path, r)
                        for c in CHECKS:
                            store[f"det_{c}"][i, j] = tri(res[c])
                store["clip_l14"][i] = score_batch(images, [r["prompt"]] * k, clip, processor)[0]
                if inc:
                    features.append(inc(images))
        if inc:
            store["inception"] = np.stack(features)
        store["ids"] = np.array([r["id"] for r in chunk])
        tmp_part = part.with_suffix(".tmp.npz")
        np.savez(tmp_part, **store)
        os.replace(tmp_part, part)
        print(f"{config}/{name} {start + len(chunk)}/{len(rows)} {time.time() - t0:.0f} s", flush=True)
    merged: dict = {}
    for p in sorted(parts.glob("*.npz")):
        with np.load(p) as f:
            for key in f.files:
                merged.setdefault(key, []).append(f[key])
    merged = {key: np.concatenate(v) for key, v in merged.items()}
    assert merged["ids"].tolist() == [r["id"] for r in rows]
    np.savez(out_dir / f"{name}.npz", **merged)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--inception", action="store_true")
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()
    guard_test_split(args.split, args.final_test)
    os.nice(10)
    run(args.config, args.split, args.k, args.inception)


if __name__ == "__main__":
    main()
