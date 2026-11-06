#!/usr/bin/env python
"""Generate images with the Core ML models on this Mac's Neural Engine. Resumable.

    python eval/generate_coreml.py --model sdxs-dreamshaper --variant A --split coco_1k --config S1-A

Writes data/gen/<config>/<split>/<id>.png and log.jsonl (per-image timing). Finished ids are skipped.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COREML, GEN, HF, SQRT_ALPHA, SQRT_ONE_MINUS_ALPHA, check_disk, done_ids, guard_test_split, latent_for_seed, pixels_from_decoder, read_split, split_name  # noqa: E402

COMPONENTS = {
    "A": ("text_pal8", "unet_fix_pal8", "taesd_fp16"),
    "FP16": ("text_fp16", "unet_fix_fp16", "taesd_fp16"),
    "STOCK": ("text_fp16", "unet_stock_fp16", "taesd_fp16"),
    "PAL6": ("text_pal6", "unet_fix_pal6", "taesd_fp16"),
}


class CoreMLGenerator:
    def __init__(self, model: str, variant: str, compute_units: str = "CPU_AND_NE", coreml_dir: Path = COREML, hf_dir: Path = HF):
        import coremltools as ct
        from transformers import CLIPTokenizer

        cu = getattr(ct.ComputeUnit, compute_units)
        compiled = coreml_dir / model / "compiled"
        t0 = time.perf_counter()
        text, unet, dec = (ct.models.CompiledMLModel(str(compiled / f"{n}.mlmodelc"), compute_units=cu) for n in COMPONENTS[variant])
        self.text, self.unet, self.dec = text, unet, dec
        self.load_s = time.perf_counter() - t0
        self.tokenizer = CLIPTokenizer.from_pretrained(hf_dir / model / "tokenizer")

    def tokenize(self, prompt: str) -> np.ndarray:
        ids = self.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np").input_ids
        return ids.astype(np.int32)

    def encode(self, prompt: str) -> np.ndarray:
        return self.text.predict({"input_ids": self.tokenize(prompt)})["emb"]

    def eps(self, z: np.ndarray, emb: np.ndarray) -> np.ndarray:
        return self.unet.predict({"sample": z, "encoder_hidden_states": emb})["eps"]

    def decode(self, z: np.ndarray, eps: np.ndarray) -> np.ndarray:
        x0 = (z - SQRT_ONE_MINUS_ALPHA * eps.astype(np.float32)) / SQRT_ALPHA
        return self.dec.predict({"latents": x0.astype(np.float32)})["image"]

    def generate(self, prompt: str, seed: int) -> tuple[np.ndarray, dict]:
        """Returns (HWC uint8 image, timings in ms). Timing covers tokenise to decoder output, no PNG write."""
        z = latent_for_seed(seed).numpy()
        t0 = time.perf_counter()
        emb = self.encode(prompt)
        t1 = time.perf_counter()
        eps = self.eps(z, emb)
        t2 = time.perf_counter()
        y = self.decode(z, eps)
        t3 = time.perf_counter()
        ms = {"text_ms": 1e3 * (t1 - t0), "unet_ms": 1e3 * (t2 - t1), "dec_ms": 1e3 * (t3 - t2), "total_ms": 1e3 * (t3 - t0)}
        return pixels_from_decoder(y[0]), ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sdxs-dreamshaper")
    ap.add_argument("--variant", choices=sorted(COMPONENTS), default="A")
    ap.add_argument("--split", required=True, help="split name in eval/splits, or a JSONL path")
    ap.add_argument("--config", required=True, help="label, e.g. S1-A; output folder name")
    ap.add_argument("--compute-units", default="CPU_AND_NE")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()
    guard_test_split(args.split, args.final_test)
    os.nice(10)

    from PIL import Image

    out = GEN / args.config / split_name(args.split)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    rows = read_split(args.split)[: args.limit]
    done = done_ids(log_path)
    rows = [r for r in rows if r["id"] not in done]
    print(f"{args.config}/{split_name(args.split)}: {len(rows)} to do, {len(done)} done, free {check_disk():.1f} GB", flush=True)
    if not rows:
        return
    gen = CoreMLGenerator(args.model, args.variant, args.compute_units)
    print(f"load {gen.load_s:.2f} s", flush=True)
    t0 = time.time()
    with open(log_path, "a") as log:
        for i, r in enumerate(rows):
            if i % 50 == 0:
                check_disk()
            img, ms = gen.generate(r["prompt"], r["seed"])
            Image.fromarray(img).save(out / f"{r['id']}.png")
            log.write(json.dumps({"id": r["id"], "seed": r["seed"], **{k: round(v, 2) for k, v in ms.items()}}) + "\n")
            log.flush()
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(rows)} {time.time() - t0:.0f} s", flush=True)
    print(f"done {len(rows)} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
