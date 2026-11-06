#!/usr/bin/env python
"""Charts from results/*.csv, for the report. Draws only what has been measured.

    python eval/charts.py

Colours follow the dataviz reference palette (validated light-mode slots 1-3). Identity is carried by direct labels and
marker shape as well as colour. The CSVs are the table view.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#8a8983"
METHOD_STYLE = {"SD-Turbo C steps": (ORANGE, "s"), "SD-Turbo best-of-C": (AQUA, "^"), "SDXS best-of-C": (BLUE, "o")}
SHIPPED = "S1-A"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlelocation": "left", "axes.titlesize": 11, "axes.axisbelow": True,
})


def read(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def scatter(points: list[dict], x_key: str, x_label: str, title: str, out: str, logx: bool = True) -> None:
    """Every config, one hue: the shipped config is blue, everything else neutral, all labelled."""
    if len(points) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for p in points:
        color = BLUE if p["config"] == SHIPPED else NEUTRAL
        ax.errorbar(p[x_key], p["clip"], yerr=[[p["clip"] - p["lo"]], [p["hi"] - p["clip"]]], fmt="none", ecolor=color, elinewidth=1.2, capsize=0)
        ax.scatter(p[x_key], p["clip"], s=64, c=color, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.annotate(p["config"], (p[x_key], p["clip"]), textcoords="offset points", xytext=(7, 5), fontsize=9, color=INK if p["config"] == SHIPPED else INK_2)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("CLIP similarity, ViT-L/14 (100 x cos)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(RESULTS / out)
    plt.close(fig)
    print(RESULTS / out)


def quality_cost() -> None:
    metrics = {r["config"]: r for r in read("generator_metrics.csv") if r["split"] == "coco_1k"}
    latency = {r["config"]: r for r in read("latency.csv")}
    pts_l, pts_s = [], []
    for config, m in metrics.items():
        if config not in latency:
            continue
        base = {"config": config, "clip": float(m["clip_l14"]), "lo": float(m["clip_l14_ci95_lo"]), "hi": float(m["clip_l14_ci95_hi"])}
        ms, size = num(latency[config].get("warm_mean_ms")), num(latency[config].get("size_MiB"))
        if ms:
            pts_l.append({**base, "x": ms})
        if size:
            pts_s.append({**base, "x": size})
    scatter(pts_l, "x", "Latency per image, ms, measured on Apple M4 (log scale)", "Quality vs latency, 1k COCO captions", "pareto_latency.png")
    scatter(pts_s, "x", "Model size on disk, MiB", "Quality vs size, 1k COCO captions", "pareto_size.png", logx=False)


def e1() -> None:
    rows = [r for r in read("e1.csv") if r["split"] == "complex_dev"]
    if not rows:
        return
    panels = [("clip_l14", "CLIP similarity, ViT-L/14"), ("count_rate", "Count check pass rate, DETR-R101"), ("position_rate", "Position check pass rate, DETR-R101")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=160)
    for ax, (key, title) in zip(axes, panels):
        for method, (color, marker) in METHOD_STYLE.items():
            pts = sorted(((int(r["C_unet_calls"]), num(r.get(key))) for r in rows if r["method"] == method and num(r.get(key)) is not None))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=2, marker=marker, markersize=7, markeredgecolor=SURFACE, label=method)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        ax.set_xlabel("UNet calls per final image")
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "e1.png")
    plt.close(fig)
    print(RESULTS / "e1.png")


def classical() -> None:
    rows = read("ablation_classical.csv")
    if not rows:
        return
    rows = sorted(rows, key=lambda r: float(r["clip_diff_vs_off"]))
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(rows) + 1.2), dpi=160)
    for i, r in enumerate(rows):
        d, lo, hi = float(r["clip_diff_vs_off"]), float(r["ci95_lo"]), float(r["ci95_hi"])
        ax.plot([lo, hi], [i, i], color=NEUTRAL, linewidth=1.5)
        ax.scatter(d, i, s=48, c=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(rows)), [f"{r['op']} {r['mode']}" for r in rows])
    ax.set_xlabel("Change in CLIP similarity vs all ops off, paired 95% CI")
    ax.set_title("Classical ops, 1k COCO captions, S1-A on Apple M4")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "ablation_classical.png")
    plt.close(fig)
    print(RESULTS / "ablation_classical.png")


if __name__ == "__main__":
    quality_cost()
    e1()
    classical()
