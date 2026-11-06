#!/usr/bin/env python
"""Streaming evaluation run: generate a chunk, extract features, delete its PNGs, repeat. Resumable.

    python eval/run.py --config S1-A --split coco_1k [--inception]
    python eval/run.py --config B1-T4 --split coco_5k --inception --from-volume kshana-eval

Keeps the first 30 images of the split as a sample (same rows for every config, nothing hand-picked).
Output: data/features/<config>/<split>.npz, data/gen/<config>/<split>/log.jsonl
A later run with --inception regenerates what it needs from the same seeds and adds the missing features.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES, GEN, check_disk, wait_for_disk, guard_test_split, read_split, split_name  # noqa: E402

KEEP_SAMPLE = 30
CHUNK = 200


def merge_parts(parts_dir: Path, rows: list[dict], out: Path) -> dict:
    parts = sorted(parts_dir.glob("*.npz"))
    merged: dict = {}
    for p in parts:
        with np.load(p) as f:
            for k in f.files:
                merged.setdefault(k, []).append(f[k])
    merged = {k: np.concatenate(v) for k, v in merged.items()}
    assert merged["ids"].tolist() == [r["id"] for r in rows], "parts do not cover the split in order"
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **merged)
    os.replace(tmp, out)
    return merged


def fetch_from_volume(volume_name: str, remote_dir: str, chunk: list[dict], img_dir: Path) -> None:
    """Copy this chunk's PNGs from a Modal volume. Reads only; nothing on the volume changes."""
    from concurrent.futures import ThreadPoolExecutor

    import modal

    volume = modal.Volume.from_name(volume_name)

    def one(row: dict) -> None:
        path = img_dir / f"{row['id']}.png"
        if not path.exists():
            with open(path.with_suffix(".part"), "wb") as f:
                for block in volume.read_file(f"{remote_dir}/{row['id']}.png"):
                    f.write(block)
            os.replace(path.with_suffix(".part"), path)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, chunk))


def run(config: str, split: str, inception: bool, from_images: bool, keep_all: bool, from_volume: str | None = None) -> None:
    from features import Extractor

    rows = read_split(split)
    name = split_name(split)
    img_dir = GEN / config / name
    img_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = FEATURES / config / f"{name}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    keep = {r["id"] for r in rows[:KEEP_SAMPLE]}
    extractor = Extractor(inception)
    gen = None
    t0 = time.time()
    with open(img_dir / "log.jsonl", "a") as log:
        for index in range(0, len(rows), CHUNK):
            chunk = rows[index : index + CHUNK]
            part = parts_dir / f"{index:06d}.npz"
            have = dict(np.load(part)) if part.exists() else {"ids": np.array([r["id"] for r in chunk])}
            need = tuple(k for k in extractor.keys if k not in have)
            if not need:
                continue
            wait_for_disk()
            if from_volume:
                fetch_from_volume(from_volume, f"gen/{config}/{name}", chunk, img_dir)
            elif not from_images:
                for r in chunk:
                    path = img_dir / f"{r['id']}.png"
                    if path.exists():
                        continue
                    if gen is None:
                        from generators import make_generator

                        gen = make_generator(config)
                        print(f"loaded generator in {gen.load_s:.2f} s", flush=True)
                    img, ms = gen.generate(r["prompt"], r["seed"])
                    Image.fromarray(img).save(path)
                    log.write(json.dumps({"id": r["id"], "seed": r["seed"], **{k: round(v, 2) for k, v in ms.items()}}) + "\n")
                log.flush()
            have.update(extractor.extract(chunk, img_dir, need))
            tmp = part.with_suffix(".tmp.npz")
            np.savez(tmp, **have)
            os.replace(tmp, part)
            if not keep_all:
                for r in chunk:
                    if r["id"] not in keep:
                        (img_dir / f"{r['id']}.png").unlink(missing_ok=True)
            print(f"{config}/{name} {index + len(chunk)}/{len(rows)} {time.time() - t0:.0f} s, free {check_disk(0):.1f} GB", flush=True)
    merged = merge_parts(parts_dir, rows, FEATURES / config / f"{name}.npz")
    print(config, name, {k: v.shape for k, v in merged.items()}, "mean clip_l14", round(float(merged["clip_l14"].mean()), 3), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--inception", action="store_true")
    ap.add_argument("--from-images", action="store_true")
    ap.add_argument("--from-volume", help="Modal volume holding gen/<config>/<split>/<id>.png")
    ap.add_argument("--keep-all", action="store_true", help="keep every PNG, for the complex-prompt scorer")
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()
    guard_test_split(args.split, args.final_test)
    os.nice(10)
    run(args.config, args.split, args.inception, args.from_images, args.keep_all, args.from_volume)


if __name__ == "__main__":
    main()
