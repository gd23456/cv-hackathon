#!/usr/bin/env python
"""Writes web/report.json from measured results. Anything not measured is left out."""
from __future__ import annotations

import csv
import json
from pathlib import Path

RESULTS = Path("results")
NAMES = {
    "S1-A": "Kshana, 8-bit",
    "S1-FP16": "Kshana, uncompressed",
    "S0.9-FP16": "SDXS 0.9",
    "B1": "SD-Turbo, 1 step",
    "B0": "SD 1.5, 25 steps",
}


def rows(name: str) -> list[dict]:
    path = RESULTS / name
    return list(csv.DictReader(path.open())) if path.exists() else []


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    metrics: dict[str, dict] = {}
    for row in rows("generator_metrics.csv"):
        entry = metrics.setdefault(row["config"], {})
        if row["split"] == "coco_1k":
            entry["clip"], entry["kid"] = number(row["clip_l14"]), number(row["kid_x1000"])
        if row["split"] == "coco_5k":
            entry["fid"] = number(row["fid"])
    latency = {row["config"]: row for row in rows("latency.csv")}

    results = []
    for config, name in NAMES.items():
        measured = metrics.get(config)
        timed = latency.get(config)
        if not measured or not timed:
            continue
        results.append({
            "name": name, "where": f"{timed['hardware']}, {timed['engine'].split(',')[0]}",
            "ms": number(timed["warm_mean_ms"]), "mb": number(timed.get("size_MiB") or timed.get("model_size_MiB")),
            "peak_mb": number(timed.get("process_maxrss_MB")), **measured,
        })

    psnr = json.loads((RESULTS / "conversion_check.json").read_text())["summary"] if (RESULTS / "conversion_check.json").exists() else {}
    compression = []
    for config, variant in (("S1-A", "A"), ("S1-FP16", "FP16")):
        if config in latency and config in metrics:
            compression.append({
                "name": NAMES[config], "mb": number(latency[config].get("size_MiB") or latency[config].get("model_size_MiB")),
                "clip": metrics[config].get("clip"), "kid": metrics[config].get("kid"),
                "psnr": psnr.get(variant, {}).get("psnr_vs_torch_mean"),
            })

    labels = {"two_object": "Two objects", "counting": "Counts", "position": "Positions", "colors": "Single colors", "color_attr": "Colors on the right objects", "overall": "Overall", "single_object": "Single objects"}
    complex_rows = [
        {"name": labels.get(row["tag"], row["tag"].replace("_", " ").capitalize()), "value": number(row["accuracy"])}
        for row in rows("complex_scores.csv")
        if row["config"] == "S1-A" and row["split"] == "complex_test" and number(row["accuracy"]) is not None
    ]

    safety = []
    holdout = rows("safety_holdout.csv")
    if holdout:
        def share(group, test):
            selected = [r for r in holdout if group(r)]
            hits = sum(1 for r in selected if test(r))
            return hits, len(selected)

        limits = {"Server": (0.5, 0.2, 0.9), "iPhone": (0.15, 0.1, None)}
        for label, (limit, skin_limit, concept) in limits.items():
            def blocked(r, limit=limit, skin_limit=skin_limit, concept=concept):
                over = float(r["marqo"]) >= (skin_limit if float(r["skin"]) >= 0.35 else limit)
                return over or (concept is not None and float(r["concept"]) >= concept)

            a, an = share(lambda r: r["group"] == "adult", blocked)
            leak, ln = share(lambda r: r["group"] != "benign" and float(r["independent"]) > 0.5, lambda r: not blocked(r))
            f, fn = share(lambda r: r["group"] == "benign", blocked)
            safety.append({
                "name": label, "blocked": f"{a}/{an}", "leaked": f"{leak}/{ln}",
                "false_block": f"{100 * f / max(1, fn):.1f}%",
            })

    gallery_path = Path("web/compare/gallery.json")
    gallery = json.loads(gallery_path.read_text()) if gallery_path.exists() else []

    Path("web/report.json").write_text(json.dumps({
        "note": "Measured by the project's evaluation scripts. Each row states where it ran, and results from different machines aren't directly comparable.",
        "results": results, "compression": compression, "complex": complex_rows, "safety": safety, "gallery": gallery,
    }, indent=1))
    print("results", len(results), "compression", len(compression), "complex", len(complex_rows), "safety", len(safety), "gallery", len(gallery))


if __name__ == "__main__":
    main()
