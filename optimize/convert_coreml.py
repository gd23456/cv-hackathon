#!/usr/bin/env python
"""Rebuild the Core ML generators from Hugging Face weights (safetensors only).

    python optimize/convert_coreml.py --model sdxs-dreamshaper
    python optimize/convert_coreml.py --model sdxs-0.9
    python optimize/convert_coreml.py --model sdxs-dreamshaper --verify .spike/ml
    python optimize/convert_coreml.py --model sdxs-0.9 --fp16-only --prune

Output: <out>/<model>/{name}.mlpackage, compiled/{name}.mlmodelc, manifest.json
Variants: A = unet_fix_pal8 + text_pal8 + taesd_fp16 ; FP16 = unet_fix_fp16 + text_fp16 + taesd_fp16
Extra components for the precision ablation: --extra unet_fix_pal6 text_pal6 unet_stock_fp16
"""
import argparse
import gc
import hashlib
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anefix import patch_silu  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPOS = {"sdxs-dreamshaper": "IDKiro/sdxs-512-dreamshaper", "sdxs-0.9": "IDKiro/sdxs-512-0.9"}
TAESD_REPO = "madebyollin/taesd"
WEIGHT_THRESHOLD = 6000  # same threshold as the spike: const tensors below this stay FP16
VARIANTS = {
    "A": ("text_pal8", "unet_fix_pal8", "taesd_fp16"),
    "FP16": ("text_fp16", "unet_fix_fp16", "taesd_fp16"),
}
# Spike file names, for --verify on sdxs-dreamshaper.
SPIKE_NAMES = {
    "text_fp16": "text_fp16",
    "text_pal8": "textx_pal8",
    "unet_fix_fp16": "unet_pade12",
    "unet_fix_pal8": "pade12_pal8",
    "taesd_fp16": "taesd_fp16",
}


def download(model: str, weights: Path) -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download

    sd = weights / model
    snapshot_download(
        REPOS[model],
        local_dir=sd,
        token=False,
        allow_patterns=[
            "unet/*.safetensors", "unet/config.json",
            "text_encoder/*.safetensors", "text_encoder/config.json",
            "tokenizer/*", "scheduler/*", "model_index.json",
        ],
    )
    td = weights / "taesd"
    snapshot_download(
        TAESD_REPO, local_dir=td, token=False,
        allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"],
    )
    return sd, td


def _convert(traced, inputs, outputs):
    import coremltools as ct

    return ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=True,
    )


def convert_text(sd: Path):
    import coremltools as ct
    from transformers import CLIPTextModel

    model = CLIPTextModel.from_pretrained(
        sd / "text_encoder", use_safetensors=True, attn_implementation="eager", torch_dtype=torch.float32
    ).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ids):
            return self.m(ids, return_dict=False)[0]

    with torch.no_grad():
        traced = torch.jit.trace(Wrap(model), torch.zeros((1, 77), dtype=torch.long), strict=False, check_trace=False)
    return _convert(
        traced,
        [ct.TensorType(name="input_ids", shape=(1, 77), dtype=np.int32)],
        [ct.TensorType(name="emb", dtype=np.float32)],
    )


def convert_unet(sd: Path, fix: bool):
    import coremltools as ct
    from diffusers import UNet2DConditionModel
    from diffusers.models.attention_processor import AttnProcessor

    unet = UNet2DConditionModel.from_pretrained(sd / "unet", use_safetensors=True, torch_dtype=torch.float32).eval()
    unet.set_attn_processor(AttnProcessor())
    if fix:
        print("patched SiLU modules:", patch_silu(unet), flush=True)
    emb_dim = unet.config.cross_attention_dim

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer("t", torch.tensor([999.0]))

        def forward(self, z, e):
            return self.m(z, self.t, encoder_hidden_states=e, return_dict=False)[0]

    example = (torch.zeros(1, 4, 64, 64), torch.zeros(1, 77, emb_dim))
    with torch.no_grad():
        traced = torch.jit.trace(Wrap(unet).eval(), example, strict=False, check_trace=False)
    return _convert(
        traced,
        [
            ct.TensorType(name="sample", shape=(1, 4, 64, 64), dtype=np.float32),
            ct.TensorType(name="encoder_hidden_states", shape=(1, 77, emb_dim), dtype=np.float32),
        ],
        [ct.TensorType(name="eps", dtype=np.float32)],
    )


def convert_taesd(td: Path):
    import coremltools as ct
    from diffusers import AutoencoderTiny

    vae = AutoencoderTiny.from_pretrained(td, use_safetensors=True, torch_dtype=torch.float32).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.d = m.decoder

        def forward(self, x):
            return self.d(x)

    with torch.no_grad():
        traced = torch.jit.trace(Wrap(vae), torch.zeros(1, 4, 64, 64), strict=False, check_trace=False)
    return _convert(
        traced,
        [ct.TensorType(name="latents", shape=(1, 4, 64, 64), dtype=np.float32)],
        [ct.TensorType(name="image", dtype=np.float32)],
    )


def palettize(src: Path, nbits: int, exclude: tuple[str, ...] = ()):
    """k-means palette. `exclude` keeps matching consts FP16: k-means breaks on inf weights in the text embeddings."""
    import coremltools as ct
    from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, get_weights_metadata, palettize_weights

    model = ct.models.MLModel(str(src), skip_model_load=True)
    per_op = {}
    if exclude:
        for name in get_weights_metadata(model, weight_threshold=WEIGHT_THRESHOLD):
            if any(x in name for x in exclude):
                per_op[name] = None
                print("kept FP16:", name, flush=True)
    config = OptimizationConfig(
        global_config=OpPalettizerConfig(mode="kmeans", nbits=nbits, weight_threshold=WEIGHT_THRESHOLD),
        op_name_configs=per_op,
    )
    return palettize_weights(model, config)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def compile_to(pkg: Path, dst: Path) -> None:
    import coremltools as ct

    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(ct.utils.compile_model(str(pkg))), dst)


def build(model: str, out: Path, weights: Path, extra: list[str], fresh: bool, fp16_only: bool, prune: bool) -> dict:
    mdir = out / model
    mdir.mkdir(parents=True, exist_ok=True)
    compiled_dir = mdir / "compiled"
    order = ["text_fp16", "taesd_fp16", "unet_fix_fp16"]
    if not fp16_only:
        order += ["text_pal8", "unet_fix_pal8"]
    order += extra
    sd, td = weights / model, weights / "taesd"
    if fresh or not all((compiled_dir / f"{n}.mlmodelc").exists() for n in order):
        sd, td = download(model, weights)

    recipes = {
        "text_fp16": lambda: convert_text(sd),
        "taesd_fp16": lambda: convert_taesd(td),
        "unet_fix_fp16": lambda: convert_unet(sd, fix=True),
        "unet_stock_fp16": lambda: convert_unet(sd, fix=False),
        "text_pal8": lambda: palettize(mdir / "text_fp16.mlpackage", 8, exclude=("embedding",)),
        "text_pal6": lambda: palettize(mdir / "text_fp16.mlpackage", 6, exclude=("embedding",)),
        "unet_fix_pal8": lambda: palettize(mdir / "unet_fix_fp16.mlpackage", 8),
        "unet_fix_pal6": lambda: palettize(mdir / "unet_fix_fp16.mlpackage", 6),
    }
    timings = {}
    for name in order:
        pkg = mdir / f"{name}.mlpackage"
        compiled = compiled_dir / f"{name}.mlmodelc"
        if compiled.exists() and not fresh:
            continue
        if fresh and pkg.exists():
            shutil.rmtree(pkg)
        if not pkg.exists():
            t0 = time.time()
            built = recipes[name]()
            built.save(str(pkg))
            del built
            gc.collect()
            timings[name] = round(time.time() - t0, 1)
            print(f"built {name} in {timings[name]} s, maxrss {maxrss_mb()} MB", flush=True)
        compile_to(pkg, compiled)
    if prune:
        for pkg in mdir.glob("*.mlpackage"):
            shutil.rmtree(pkg)  # the compiled copy is what runs; rebuild with this script if a source is needed
        for sub in ("unet", "text_encoder"):
            if (weights / model / sub).exists():
                shutil.rmtree(weights / model / sub)  # keep tokenizer/, needed at generation time

    components = {
        p.name[: -len(".mlmodelc")]: {"compiled_bytes": tree_bytes(p), "weight_sha256": sha256(p / "weights/weight.bin")}
        for p in sorted(compiled_dir.glob("*.mlmodelc"))
    }
    variants = {
        v: {"components": list(names), "compiled_MB": round(sum(components[n]["compiled_bytes"] for n in names) / 1e6, 1),
            "compiled_MiB": round(sum(components[n]["compiled_bytes"] for n in names) / 2**20, 1)}
        for v, names in VARIANTS.items()
        if all(n in components for n in names)
    }
    manifest = {
        "model": model,
        "source_repos": [REPOS[model], TAESD_REPO],
        "components": components,
        "variants": variants,
        "conversion_maxrss_MB": maxrss_mb(),
        "build_seconds": timings,
    }
    (mdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def maxrss_mb() -> int:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576)


def verify_hashes(model: str, out: Path, spike: Path) -> bool:
    """Compare compiled weights and program of the rebuild with the spike's, byte for byte."""
    ok = True
    for ours, theirs in SPIKE_NAMES.items():
        a_dir, b_dir = out / model / "compiled" / f"{ours}.mlmodelc", spike / "compiled" / f"{theirs}.mlmodelc"
        same = all(sha256(a_dir / f) == sha256(b_dir / f) for f in ("weights/weight.bin", "model.mil"))
        ok &= same
        print(f"{ours:16s} vs spike {theirs:14s} {'IDENTICAL' if same else 'DIFFERENT'}", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(REPOS), default="sdxs-dreamshaper")
    ap.add_argument("--out", type=Path, default=ROOT / "data/coreml")
    ap.add_argument("--weights", type=Path, default=ROOT / "data/hf")
    ap.add_argument("--extra", nargs="*", default=[], choices=["unet_fix_pal6", "text_pal6", "unet_stock_fp16"])
    ap.add_argument("--fresh", action="store_true", help="rebuild even if outputs exist")
    ap.add_argument("--fp16-only", action="store_true", help="skip the palettized components")
    ap.add_argument("--prune", action="store_true", help="drop .mlpackage copies and downloaded weights once compiled")
    ap.add_argument("--verify", type=Path, help="spike ml/ directory: compare weight.bin hashes")
    args = ap.parse_args()

    manifest = build(args.model, args.out, args.weights, args.extra, args.fresh, args.fp16_only, args.prune)
    print(json.dumps(manifest["variants"], indent=1))
    if args.verify:
        sys.exit(0 if verify_hashes(args.model, args.out, args.verify) else 1)


if __name__ == "__main__":
    main()
