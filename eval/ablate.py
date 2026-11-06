#!/usr/bin/env python
"""Ablation and E1 tables from stored features and candidate stores. Only what has been measured appears.

    python eval/ablate.py

results/ablation_decoder_precision.csv  decoder and precision rows, paired against S1-A
results/ablation_n_selector.csv         best-of-N for N in 1,2,4,8 and each selector
results/ablation_classical.csv          each classical op off, always, gated, paired against off
results/e1.csv                          C UNet calls: SD-Turbo C steps, SD-Turbo best-of-C, SDXS best-of-C

Best-of-N is reported on ViT-L/14 and DETR-R101 only: the selector's own CLIP is ViT-B/32. FID is never compared across N.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import best_index  # noqa: E402
from common import DATA, FEATURES, RESULTS  # noqa: E402
from configs import CONFIGS, config_info  # noqa: E402
from metrics import paired_bootstrap, upsert_csv  # noqa: E402

NS = (1, 2, 4, 8)
CLASSICAL_OPS = ("inpaint", "gray_world", "clahe", "guided_filter", "unsharp")
DEV_SPLITS = ("complex_dev", "coco_1k")


def load_candidates(config: str, split: str) -> dict | None:
    path = DATA / "candidates" / config / f"{split}.npz"
    if not path.exists():
        return None
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def rate(values: np.ndarray) -> tuple[float | str, int]:
    """Mean over applicable prompts (-1 means the check does not apply)."""
    usable = values[values >= 0]
    return (round(float(usable.mean()), 4), len(usable)) if len(usable) else ("", 0)


def selected_row(store: dict, selector: str, n: int) -> tuple[dict, np.ndarray]:
    idx = best_index(store[f"sel_{selector}"], n)
    rows = np.arange(len(idx))
    l14 = np.maximum(store["clip_l14"][rows, idx], 0)
    row: dict = {"clip_l14": round(float(l14.mean()), 3)}
    for check in ("correct", "presence", "count", "position", "color"):
        key = f"det_{check}"
        if key in store:
            row[f"{check}_rate"], row[f"{check}_n"] = rate(store[key][rows, idx])
    return row, l14


def n_selector_table() -> list[dict]:
    out = []
    for config in ("S1-A", "B1"):
        for split in DEV_SPLITS:
            store = load_candidates(config, split)
            if store is None:
                continue
            _, base = selected_row(store, "both", 1)
            for selector in ("clip", "detector", "both"):
                for n in NS:
                    if n > store["sel_both"].shape[1]:
                        continue
                    row, l14 = selected_row(store, selector, n)
                    paired = paired_bootstrap(l14, base)
                    out.append({"config": config, "split": split, "selector": selector, "N": n, "prompts": len(l14), **row,
                                "clip_diff_vs_N1": round(paired["diff"], 3), "ci95_lo": round(paired["ci_lo"], 3), "ci95_hi": round(paired["ci_hi"], 3),
                                "hardware": CONFIGS[config]["hardware"], "engine": CONFIGS[config]["engine"],
                                "selector_engine": "server verifier: YOLOX-S + ViT-B/32 FP16, onnxruntime CPU"})
    return out


def e1_table() -> list[dict]:
    """C UNet calls per final image. Selection cost is not counted as UNet calls and is listed separately in latency."""
    out = []
    for split in DEV_SPLITS:
        for c in NS:
            for label, config, kind in (("SD-Turbo C steps", "B1" if c == 1 else f"B1-{c}step", "steps"),
                                        ("SD-Turbo best-of-C", "B1", "best"), ("SDXS best-of-C", "S1-A", "best")):
                if kind == "steps":
                    path = FEATURES / config / f"{split}.npz"
                    if not path.exists():
                        continue
                    with np.load(path) as f:
                        l14 = np.maximum(f["clip_l14"], 0)
                    row: dict = {"clip_l14": round(float(l14.mean()), 3)}
                    scores = DATA / "scores" / config / f"{split}.csv"
                    if scores.exists():
                        with open(scores, newline="") as sf:
                            rows = list(csv.DictReader(sf))
                        for check in ("correct", "presence", "count", "position", "color"):
                            vals = [r[check] == "True" for r in rows if r.get(check) in ("True", "False")]
                            if vals:
                                row[f"{check}_rate"], row[f"{check}_n"] = round(float(np.mean(vals)), 4), len(vals)
                else:
                    store = load_candidates(config, split)
                    if store is None or c > store["sel_both"].shape[1]:
                        continue
                    row, l14 = selected_row(store, "both", c)
                lo, hi = np.percentile(np.random.default_rng(0).choice(l14, (2000, len(l14))).mean(1), [2.5, 97.5])
                out.append({"split": split, "C_unet_calls": c, "method": label, "config": config, "prompts": len(l14), **row,
                            "clip_ci95_lo": round(float(lo), 3), "clip_ci95_hi": round(float(hi), 3),
                            "hardware": CONFIGS[config]["hardware"], "engine": CONFIGS[config]["engine"]})
    return out


def feature_table(names: list[str], base: str, split: str = "coco_1k") -> list[dict]:
    """CLIP against a paired base config, from stored per-image features; KID and CMMD come from generator_metrics.csv."""
    base_path = FEATURES / base / f"{split}.npz"
    if not base_path.exists():
        return []
    with np.load(base_path) as f:
        base_score, base_ids = f["clip_l14"], f["ids"].tolist()
    extras = {}
    metrics_csv = RESULTS / "generator_metrics.csv"
    if metrics_csv.exists():
        with open(metrics_csv, newline="") as f:
            extras = {r["config"]: r for r in csv.DictReader(f) if r["split"] == split}
    out = []
    for name in names:
        path = FEATURES / name / f"{split}.npz"
        if not path.exists():
            continue
        with np.load(path) as f:
            score, ids = f["clip_l14"], f["ids"].tolist()
        assert ids == base_ids, "configs are not paired"
        p = paired_bootstrap(np.maximum(score, 0), np.maximum(base_score, 0))
        out.append({"config": name, "base": base, "split": split, "n": len(score), "clip_l14": round(p["mean_a"], 3),
                    "clip_diff": round(p["diff"], 3), "ci95_lo": round(p["ci_lo"], 3), "ci95_hi": round(p["ci_hi"], 3),
                    "kid_x1000": extras.get(name, {}).get("kid_x1000", ""), "cmmd_l14_x1000": extras.get(name, {}).get("cmmd_l14_x1000", ""),
                    "hardware": config_info(name)["hardware"], "engine": config_info(name)["engine"], "precision": config_info(name)["precision"], "decoder": config_info(name)["decoder"]})
    return out


def classical_table() -> list[dict]:
    base = "S1-A"  # every op off is the plain generator, same seeds
    names = [f"S1-A+{op}={mode}" for op in CLASSICAL_OPS for mode in ("always", "gated")]
    out = []
    for row in feature_table(names, base):
        op, mode = row["config"].split("+")[1].split("=")
        log = DATA / "gen" / row["config"] / "coco_1k/log.jsonl"
        applied, ms = [], []
        if log.exists():
            with open(log) as f:
                for line in f:
                    rec = json.loads(line)
                    applied.append(rec.get("applied", 0.0))
                    ms.append(rec.get("classical_ms", 0.0))
        out.append({"op": op, "mode": mode, "n": row["n"], "clip_l14": row["clip_l14"], "clip_diff_vs_off": row["clip_diff"], "ci95_lo": row["ci95_lo"], "ci95_hi": row["ci95_hi"],
                    "kid_x1000": row["kid_x1000"], "cmmd_l14_x1000": row["cmmd_l14_x1000"], "applied_rate": round(float(np.mean(applied)), 3) if applied else "",
                    "op_ms_mean": round(float(np.mean(ms)), 2) if ms else "", "hardware": row["hardware"], "engine": row["engine"] + "; ops on CPU"})
    return out


def main() -> None:
    decoder_precision = feature_table(["S1-FP32", "S1-FP16", "S1-PAL6", "S1-STOCK", "S1-VAE", "S0.9-FP16"], "S1-A")
    if decoder_precision:
        upsert_csv(RESULTS / "ablation_decoder_precision.csv", decoder_precision, ("config", "split"))
    classical = classical_table()
    if classical:
        upsert_csv(RESULTS / "ablation_classical.csv", classical, ("op", "mode"))
    print(f"decoder/precision rows {len(decoder_precision)}, classical rows {len(classical)}")
    n_sel = n_selector_table()
    if n_sel:
        upsert_csv(RESULTS / "ablation_n_selector.csv", n_sel, ("config", "split", "selector", "N"))
    e1 = e1_table()
    if e1:
        upsert_csv(RESULTS / "e1.csv", e1, ("split", "C_unet_calls", "method"))
    print(f"n_selector rows {len(n_sel)}, e1 rows {len(e1)}")


if __name__ == "__main__":
    main()
