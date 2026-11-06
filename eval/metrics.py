#!/usr/bin/env python
"""Metric tables from stored features. Same seeds and prompts across configs, so CIs are paired.

    python eval/metrics.py --split coco_1k --configs S1-A S1-FP16 S0.9-FP16 B1 --baseline S1-A

Writes results/generator_metrics.csv (one row per config and split) and results/paired_bootstrap.csv.
FID is computed only for 5,000 images against the 5,000-image reference.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES, RESULTS, guard_test_split  # noqa: E402
from configs import config_info  # noqa: E402

FID_N = 5000
BOOTSTRAP = 10000
PROTOCOL = "COCO val2017 captions; 5k reference center-crop 512 Lanczos; clean-fid clean mode; PNG only"


def clip_headline(score: np.ndarray) -> np.ndarray:
    """max(100*cos, 0) per image."""
    return np.maximum(score, 0.0)


def bootstrap_ci(x: np.ndarray, n: int = BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = BOOTSTRAP, seed: int = 0) -> dict:
    """a and b are per-image values from the same (prompt, seed) pairs, in the same order."""
    assert len(a) == len(b)
    d = a - b
    lo, hi = bootstrap_ci(d, n, seed)
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def fid(gen: np.ndarray, ref: np.ndarray) -> float:
    if len(gen) != FID_N or len(ref) != FID_N:
        raise ValueError(f"FID is only reported at {FID_N} vs {FID_N}, got {len(gen)} vs {len(ref)}")
    from cleanfid.fid import frechet_distance

    return float(frechet_distance(gen.mean(0), np.cov(gen, rowvar=False), ref.mean(0), np.cov(ref, rowvar=False)))


def kid_x1000(gen: np.ndarray, ref: np.ndarray, seed: int = 0) -> float:
    """clean-fid kernel_distance (100 subsets of up to 1000), times 1000."""
    from cleanfid.fid import kernel_distance

    np.random.seed(seed)
    return 1000.0 * float(kernel_distance(gen, ref))


def cmmd_x1000(gen: np.ndarray, ref: np.ndarray, sigma: float = 10.0) -> float:
    """CMMD formula (RBF, sigma 10, unit embeddings) on ViT-L/14 224 embeddings. The paper uses L/14 336."""

    def mean_k(x: np.ndarray, y: np.ndarray) -> float:
        y_sq = (y**2).sum(1)
        total = 0.0
        for start in range(0, len(x), 1000):
            part = x[start : start + 1000]
            d2 = (part**2).sum(1, keepdims=True) - 2 * part @ y.T + y_sq
            total += np.exp(-d2 / (2 * sigma**2)).sum()
        return float(total / (len(x) * len(y)))

    return 1000.0 * (mean_k(gen, gen) + mean_k(ref, ref) - 2 * mean_k(gen, ref))


def load(config: str, split: str) -> dict:
    with np.load(FEATURES / config / f"{split}.npz") as f:
        return {k: f[k] for k in f.files}


def load_reference() -> dict:
    with np.load(FEATURES / "reference/coco5k.npz") as f:
        return {k: f[k] for k in f.files}


def upsert_csv(path: Path, rows: list[dict], keys: tuple[str, ...]) -> None:
    """Replace rows with the same key columns, keep the rest."""
    old = []
    if path.exists():
        with open(path, newline="") as f:
            old = list(csv.DictReader(f))
    new_keys = {tuple(str(r[k]) for k in keys) for r in rows}
    merged = [r for r in old if tuple(r[k] for k in keys) not in new_keys] + rows
    fields = list(dict.fromkeys(k for r in merged for k in r))
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(merged)


def summarize(config: str, split: str, feats: dict, ref: dict | None) -> dict:
    cfg = config_info(config)
    l14 = clip_headline(feats["clip_l14"])
    lo, hi = bootstrap_ci(l14)
    row = {
        "config": config, "split": split, "n": len(l14),
        "clip_l14": round(float(l14.mean()), 3), "clip_l14_ci95_lo": round(lo, 3), "clip_l14_ci95_hi": round(hi, 3),
        "clip_b32_cos": round(float(feats["clip_b32"].mean() / 100), 4),
        "fid": "", "kid_x1000": "", "cmmd_l14_x1000": "",
        "hardware": cfg["hardware"], "engine": cfg["engine"], "precision": cfg["precision"], "steps": cfg["steps"], "decoder": cfg["decoder"],
        "protocol": PROTOCOL,
    }
    if ref is not None:
        row["cmmd_l14_x1000"] = round(cmmd_x1000(feats["emb_l14"], ref["emb_l14"]), 3)
        if "inception" in feats and "inception" in ref:
            row["kid_x1000"] = round(kid_x1000(feats["inception"], ref["inception"]), 3)
            if len(l14) == FID_N:
                row["fid"] = round(fid(feats["inception"], ref["inception"]), 3)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--baseline", help="config every other config is paired against")
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()
    guard_test_split(args.split, args.final_test)
    ref_path = FEATURES / "reference/coco5k.npz"
    ref = load_reference() if args.split.startswith("coco") and ref_path.exists() else None
    feats = {c: load(c, args.split) for c in args.configs}
    rows = [summarize(c, args.split, feats[c], ref) for c in args.configs]
    upsert_csv(RESULTS / "generator_metrics.csv", rows, ("config", "split"))
    for r in rows:
        print({k: r[k] for k in ("config", "n", "clip_l14", "clip_l14_ci95_lo", "clip_l14_ci95_hi", "clip_b32_cos", "fid", "kid_x1000", "cmmd_l14_x1000")})
    if args.baseline:
        pairs = []
        base = feats[args.baseline]
        for c in args.configs:
            if c == args.baseline:
                continue
            assert feats[c]["ids"].tolist() == base["ids"].tolist(), "configs are not paired"
            for key, scale in (("clip_l14", 1.0), ("clip_b32", 0.01)):
                p = paired_bootstrap(feats[c][key] * scale, base[key] * scale)
                pairs.append({"split": args.split, "metric": key, "a": c, "b": args.baseline, "n": len(base[key]),
                              **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}})
        upsert_csv(RESULTS / "paired_bootstrap.csv", pairs, ("split", "metric", "a", "b"))
        for p in pairs:
            print(p)


if __name__ == "__main__":
    main()
