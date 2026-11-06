#!/usr/bin/env python
"""Latency and memory for one config, in fresh processes, on this Mac. Not iPhone numbers.

    python eval/bench_coreml.py --config S1-A --warm 200 --cold 3
    python eval/bench_coreml.py --config B1 --warm 30 --cold 2

Warm: per-image time over fixed prompts after a discard. Cold: process start to first image, repeated.
Memory: process max RSS, which excludes the system Neural Engine service, and the service's RSS for context.
Runs at normal priority so latency is not distorted by nice. Writes results/latency.csv.
"""
import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COREML, HF, RESULTS, hardware_label, read_split  # noqa: E402
from configs import CONFIGS  # noqa: E402

DISCARD = 5


def service_rss_mb() -> float | None:
    out = subprocess.run(["ps", "-axo", "rss,comm"], capture_output=True, text=True).stdout
    values = [int(line.split()[0]) for line in out.splitlines()[1:] if line.split()[-1].endswith("/aned")]
    return round(sum(values) / 1024, 1) if values else None


def child(config: str, warm: int) -> None:
    from generators import make_generator

    t0 = time.perf_counter()
    gen = make_generator(config)
    load_s = time.perf_counter() - t0
    rows = read_split("coco_1k")
    first_ms = gen.generate(rows[0]["prompt"], rows[0]["seed"])[1]["total_ms"]
    runs = [gen.generate(rows[i % len(rows)]["prompt"], rows[i % len(rows)]["seed"])[1] for i in range(1, warm + DISCARD + 1)][DISCARD:]
    print("RESULT " + json.dumps({
        "load_s": load_s, "first_ms": first_ms, "cold_to_image_s": load_s + first_ms / 1e3, "runs": runs,
        "maxrss_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, "aned_rss_MB": service_rss_mb(),
    }), flush=True)


def size_mib(config: str) -> float | str:
    cfg = CONFIGS[config]
    if cfg["kind"] == "coreml":
        from generate_coreml import COMPONENTS

        manifest = json.loads((COREML / cfg["model"] / "manifest.json").read_text())
        return round(sum(manifest["components"][n]["compiled_bytes"] for n in COMPONENTS[cfg["variant"]]) / 2**20, 1)
    if cfg["kind"] == "diffusers":
        folder = HF / cfg["repo"].split("/")[-1]
        return round(sum(f.stat().st_size for f in folder.rglob("*.safetensors")) / 2**20, 1)
    return ""


def mean_of(runs: list[dict], key: str) -> float | str:
    values = [r[key] for r in runs if key in r]
    return round(float(np.mean(values)), 1) if values else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--warm", type=int, default=200)
    ap.add_argument("--cold", type=int, default=3)
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args()
    if args.child:
        child(args.config, args.warm)
        return

    def spawn(warm: int) -> dict:
        out = subprocess.run([sys.executable, __file__, "--config", args.config, "--warm", str(warm), "--child"], capture_output=True, text=True, check=True).stdout
        return json.loads(next(line for line in out.splitlines() if line.startswith("RESULT"))[7:])

    res = spawn(args.warm)
    cold = [res["cold_to_image_s"]] + [spawn(1)["cold_to_image_s"] for _ in range(args.cold - 1)]
    total = np.array([r["total_ms"] for r in res["runs"]])
    from metrics import upsert_csv

    row = {
        "config": args.config, "hardware": hardware_label(), "engine": CONFIGS[args.config]["engine"], "n_warm": len(total),
        "warm_mean_ms": round(float(total.mean()), 1), "warm_p50_ms": round(float(np.percentile(total, 50)), 1), "warm_p95_ms": round(float(np.percentile(total, 95)), 1),
        "text_ms": mean_of(res["runs"], "text_ms"), "unet_ms": mean_of(res["runs"], "unet_ms"), "decoder_ms": mean_of(res["runs"], "dec_ms"),
        "cold_to_first_image_s_mean": round(float(np.mean(cold)), 2), "n_cold": len(cold),
        "process_maxrss_MB": round(res["maxrss_MB"], 0), "aned_service_rss_MB": res["aned_rss_MB"], "size_MiB": size_mib(args.config),
        "scope": "full generation, no PNG write; cold is model load plus first image, with OS caches warm; memory is process max RSS, excluding the Neural Engine service",
    }
    upsert_csv(RESULTS / "latency.csv", [row], ("config", "hardware", "engine"))
    print(row)


if __name__ == "__main__":
    main()
