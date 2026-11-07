#!/usr/bin/env python
"""Numbers section of eval_P3_T78.md, from results/*.csv only. Unmeasured items stay as TODO; no prose is generated.

    python eval/write_eval_md.py

Writes results/eval_P3_T78.md. Line 1 is exactly the USNs. Commentary is written by the human team.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COREML, RESULTS  # noqa: E402

USNS = "1RUA24CSE0146,1RUA24CSE0201,1RUA24CSE0139,1RUA24CSE0196"
TODO = "TODO: not measured yet"


def read(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def table(headers: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) if c not in (None, "") else "–" for c in r) + " |" for r in rows]
    return lines + [""]


def ci(r: dict) -> str:
    return f"{r['clip_l14']} [{r['clip_l14_ci95_lo']}, {r['clip_l14_ci95_hi']}]"


def is_flickr(r: dict) -> bool:
    return r["split"].startswith("flickr")


def clip_section(metrics: list[dict]) -> list[str]:
    rows = [[r["config"], r["split"], r["n"], ci(r), r["clip_b32_cos"], r["hardware"], r["engine"]] for r in metrics if not is_flickr(r)]
    flickr = [[r["config"], r["split"], r["n"], ci(r), r["clip_b32_cos"], r["hardware"], r["engine"]] for r in metrics if is_flickr(r)]
    flickr_lines = [f"Flickr30K, separate protocol: {TODO}", ""] if not flickr else [
        "Flickr30K, separate protocol: official 1k test captions, own real-image reference. Not comparable with the COCO rows.", "",
        *table(["Config", "Prompts", "n", "CLIP ViT-L/14", "CLIP ViT-B/32 cosine", "Hardware", "Engine"], flickr)]
    return ["## 1. CLIP similarity", "",
            "Headline ViT-L/14 via torchmetrics, max(100 x cos, 0), against the original prompt, with bootstrap 95% CI. ViT-B/32 is raw cosine.", "",
            *table(["Config", "Prompts", "n", "CLIP ViT-L/14", "CLIP ViT-B/32 cosine", "Hardware", "Engine"], rows),
            *flickr_lines,
            "iPhone 15: quality is measured on the M4 with the same Core ML models, and wasn't measured separately on the phone.", ""]


def fid_section(metrics: list[dict]) -> list[str]:
    five = [r for r in metrics if r["fid"]]
    rows = [[r["config"], r["n"], r["fid"], r["hardware"], r["engine"]] for r in five]
    kid = [r for r in metrics if (r["kid_x1000"] or r["cmmd_l14_x1000"]) and not is_flickr(r)]
    flickr_kid = [r for r in metrics if r["kid_x1000"] and is_flickr(r)]
    lines = ["## 2. FID", "", "clean-fid clean mode, PNG, 5,000 generated vs the same 5,000 COCO val2017 references (center crop, 512 Lanczos).", ""]
    lines += table(["Config", "n", "FID", "Hardware", "Engine"], rows) if rows else [f"FID at 5k: {TODO}", ""]
    if kid:
        lines += ["KID and CMMD are for ablations; FID is not reported below 5k. CMMD uses ViT-L/14 at 224, the paper uses 336.", ""]
        lines += table(["Config", "Split", "n", "KID x 1000", "CMMD x 1000"], [[r["config"], r["split"], r["n"], r["kid_x1000"], r["cmmd_l14_x1000"]] for r in kid])
    if flickr_kid:
        lines += ["Flickr30K, separate protocol: KID against 1,000 Flickr30K test images, preprocessed the same way. No FID below 5k. Not comparable with the COCO rows.", ""]
        lines += table(["Config", "Split", "n", "KID x 1000"], [[r["config"], r["split"], r["n"], r["kid_x1000"]] for r in flickr_kid])
    return lines


def iphone(*kinds: str) -> list[str]:
    rows = [r for r in read("iphone_measured.csv") if r["kind"] in kinds]
    if not rows:
        return [f"iPhone 15: {TODO} (from the app).", ""]
    return table(["iPhone measure", "Value", "Device", "OS", "Config", "Conditions", "Source"],
                 [[r["measure"], r["value"], r["device"], r["os"], r["config"], r["conditions"], r["source"]] for r in rows])


def latency_section(latency: list[dict]) -> list[str]:
    rows = [[r["config"], r["hardware"], r["engine"], r["n_warm"], r["warm_mean_ms"], r["warm_p50_ms"], r["warm_p95_ms"], r["text_ms"], r["unet_ms"], r["decoder_ms"], r["cold_to_first_image_s_mean"]] for r in latency]
    return ["## 3. Image generation latency", "",
            *table(["Config", "Hardware", "Engine", "Warm n", "Mean ms", "p50 ms", "p95 ms", "Text ms", "UNet ms", "Decoder ms", "Cold to first image s"], rows),
            *iphone("latency", "load", "failure"), "Hosting: no Modal run. The website runs on the Apple M4 through a Cloudflare Quick Tunnel; first image about 1.4 s on the idle M4 with 4 threads.", ""]


def memory_section(latency: list[dict]) -> list[str]:
    rows = [[r["config"], r["hardware"], r["engine"], r["process_maxrss_MB"], r["aned_service_rss_MB"]] for r in latency]
    return ["## 4. Peak memory consumption", "",
            *table(["Config", "Hardware", "Engine", "Process max RSS MB (excludes Neural Engine service)", "Neural Engine service RSS MB"], rows),
            *iphone("memory")]


def baseline_sizes() -> list[str]:
    from common import HF

    rows = []
    for label, folder in (("B0, SD 1.5 fp16 weights", "stable-diffusion-v1-5"), ("B1, SD-Turbo fp16 weights", "sd-turbo")):
        files = [f for f in (HF / folder).rglob("*.safetensors")]
        if files:
            size = sum(f.stat().st_size for f in files)
            rows.append([label, round(size / 2**20, 1), round(size / 1e6, 1)])
    return table(["Baseline", "MiB", "MB"], rows) if rows else [f"Baselines B0 and B1 on disk: {TODO}"]


def size_section() -> list[str]:
    import json

    rows = []
    for manifest_path in sorted(COREML.glob("*/manifest.json")):
        m = json.loads(manifest_path.read_text())
        for name, c in m["components"].items():
            rows.append([m["model"], name, round(c["compiled_bytes"] / 2**20, 1), round(c["compiled_bytes"] / 1e6, 1)])
        for variant, v in m["variants"].items():
            rows.append([m["model"], f"**variant {variant} total**", v["compiled_MiB"], v["compiled_MB"]])
    return ["## 5. Model size", "", "Compiled Core ML, generator only.", "",
            *table(["Model", "Component", "MiB", "MB"], rows),
            "Adding the checks and safety models on the iPhone: 122.6 MiB more (compiled, measured with du).", "", *baseline_sizes(), ""]


def quality_cost_section(metrics: list[dict], latency: list[dict]) -> list[str]:
    by_config = {r["config"]: r for r in latency}
    rows = [[r["config"], ci(r), by_config.get(r["config"], {}).get("warm_mean_ms"), by_config.get(r["config"], {}).get("size_MiB"), r["steps"], r["hardware"]]
            for r in metrics if r["split"] == "coco_1k"]
    lines = ["## 6. Quality versus computational cost", "",
             *table(["Config", "CLIP ViT-L/14", "Mean latency ms", "Size MiB", "UNet calls", "Hardware"], rows)]
    for name in ("pareto_latency.png", "pareto_size.png"):
        lines += [f"![{name}](results/{name})", ""] if (RESULTS / name).exists() else [f"{name}: {TODO}", ""]
    search = read("search_benefit.csv")
    if search:
        lines += ["Search: best of 8 against one image, 654 held-out complex prompts, Apple M4 Core ML, original wording. The runtime score picks; DETR-R101 and CLIP L/14 grade. Percentages except the CLIP row.", ""]
        lines += table(["Check", "n", "One image", "Best of 8", "Difference", "95% CI"],
                       [[r["label"], r["n"], r["one"], r["best_of_8"], r["diff"], f"[{r['ci95_lo']}, {r['ci95_hi']}]"] for r in search])
    else:
        lines += [f"Search: {TODO}", ""]
    lines += ["Classical steps on against off: not measured.", ""]
    complex_rows = read("complex_scores.csv")
    if complex_rows:
        lines += ["Complex prompts, GenEval rules with DETR-R101 boxes. Not comparable with published GenEval numbers.", ""]
        lines += table(["Config", "Split", "Tag", "Accuracy", "n"], [[r["config"], r["split"], r["tag"], r["accuracy"], r["n"]] for r in complex_rows])
    else:
        lines += [f"Complex prompts: {TODO}", ""]
    lines += ["Hand labels: not done.", ""]
    return lines


def main() -> None:
    metrics, latency = read("generator_metrics.csv"), read("latency.csv")
    lines = [USNS, "", *clip_section(metrics), *fid_section(metrics), *latency_section(latency), *memory_section(latency), *size_section(), *quality_cost_section(metrics, latency)]
    out = RESULTS.parent / "eval_P3_T78.md"
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
