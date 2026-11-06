#!/usr/bin/env python
"""Rebuilt models vs spike models vs PyTorch, on the spike's four prompts and seeds. Needs ANE and HF weights.

    python tests/eval/check_coreml_vs_spike.py [--spike .spike/ml]

Prints PSNR. PLAN section 4 recorded 28.8 dB (variant A) and 38.3 dB (FP16) against PyTorch.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from common import HF, RESULTS, latent_for_seed, pixels_from_decoder  # noqa: E402
from generate_coreml import COMPONENTS, CoreMLGenerator  # noqa: E402

PROMPTS = [
    "a red apple to the left of a blue cup on a wooden table",
    "a close-up picture of an old man standing in the rain",
    "a golden retriever puppy sitting on green grass, sunny day",
    "a small cabin beside a lake at sunset, mountains in the background",
]
SEEDS = [42, 43, 44, 45]
SPIKE_NAMES = {"text_pal8": "textx_pal8", "unet_fix_pal8": "pade12_pal8", "text_fp16": "text_fp16", "unet_fix_fp16": "unet_pade12", "taesd_fp16": "taesd_fp16"}
ALPHA = 0.00466009508818388


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(float) - b.astype(float)) ** 2).mean()
    return 99.0 if mse == 0 else float(10 * np.log10(255**2 / mse))


def torch_refs() -> list[np.ndarray]:
    from diffusers import AutoencoderTiny, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    sd = HF / "sdxs-dreamshaper"
    unet = UNet2DConditionModel.from_pretrained(sd / "unet", use_safetensors=True, torch_dtype=torch.float32).eval()
    text = CLIPTextModel.from_pretrained(sd / "text_encoder", use_safetensors=True, torch_dtype=torch.float32).eval()
    tok = CLIPTokenizer.from_pretrained(sd / "tokenizer")
    vae = AutoencoderTiny.from_pretrained(HF / "taesd", use_safetensors=True, torch_dtype=torch.float32).eval()
    refs = []
    a = torch.tensor(ALPHA)
    with torch.no_grad():
        for prompt, seed in zip(PROMPTS, SEEDS):
            ids = tok(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
            z = latent_for_seed(seed)
            eps = unet(z, 999, encoder_hidden_states=text(ids)[0]).sample
            x0 = (z - (1 - a).sqrt() * eps) / a.sqrt()
            refs.append(pixels_from_decoder(vae.decoder(x0)[0].numpy()))
    return refs


def spike_generator(spike: Path, variant: str) -> CoreMLGenerator:
    import coremltools as ct

    gen = CoreMLGenerator.__new__(CoreMLGenerator)
    cu = ct.ComputeUnit.CPU_AND_NE
    gen.text, gen.unet, gen.dec = (
        ct.models.CompiledMLModel(str(spike / "compiled" / f"{SPIKE_NAMES[n]}.mlmodelc"), compute_units=cu) for n in COMPONENTS[variant]
    )
    from transformers import CLIPTokenizer

    gen.tokenizer = CLIPTokenizer.from_pretrained(HF / "sdxs-dreamshaper" / "tokenizer")
    return gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spike", type=Path, default=ROOT / ".spike/ml")
    args = ap.parse_args()
    refs = torch_refs()
    rows = []
    for variant in ("A", "FP16"):
        ours = CoreMLGenerator("sdxs-dreamshaper", variant)
        spike = spike_generator(args.spike, variant)
        for i, (prompt, seed) in enumerate(zip(PROMPTS, SEEDS)):
            a, _ = ours.generate(prompt, seed)
            b, _ = spike.generate(prompt, seed)
            rows.append({"variant": variant, "seed": seed, "psnr_vs_spike": round(psnr(a, b), 2), "psnr_vs_torch": round(psnr(a, refs[i]), 2),
                         "md5": hashlib.md5(a.tobytes()).hexdigest()[:10]})
            print(rows[-1], flush=True)
    summary = {
        v: {
            "psnr_vs_spike_mean": round(float(np.mean([r["psnr_vs_spike"] for r in rows if r["variant"] == v])), 2),
            "psnr_vs_torch_mean": round(float(np.mean([r["psnr_vs_torch"] for r in rows if r["variant"] == v])), 2),
        }
        for v in ("A", "FP16")
    }
    print(summary)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "conversion_check.json").write_text(json.dumps({"engine": "Core ML CPU_AND_NE on M4; PyTorch fp32 CPU reference", "summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
