#!/usr/bin/env python
"""Export SDXS-512-DreamShaper + TAESD and check the result against PyTorch.

Text encoder, UNet and TAESD decoder: ONNX, because ONNX Runtime needs far less memory
(see kshana/generator.py). The UNet is also written as FP16 OpenVINO IR so
KSHANA_UNET_BACKEND can choose. Also fetches the verifier's ONNX models.

    python optimize/export_openvino.py --out server/models
    python optimize/export_openvino.py --out server/models \
        --sdxs /path/to/sdxs --taesd /path/to/taesd        # local weights
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import openvino as ov
import torch

SQRT_ONE_MINUS_ALPHA = 0.99766725
SQRT_ALPHA = 0.06826489
PROMPTS = [
    "a red apple to the left of a blue cup on a wooden table",
    "a close-up picture of an old man standing in the rain",
    "a golden retriever puppy sitting on green grass, sunny day",
    "a small cabin beside a lake at sunset, mountains in the background",
]
SEEDS = [42, 43, 44, 45]

# Verifier models, see kshana/LICENSES.md.
YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
CLIP_REPO = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"
CLIP_FILES = {
    "clip_vision.onnx": "onnx/vision_model_fp16.onnx",
    "clip_text.onnx": "onnx/text_model_fp16.onnx",
    "clip_tokenizer.json": "tokenizer.json",
}


class TextWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return self.m(input_ids, return_dict=False)[0]


class UNetWrap(torch.nn.Module):
    """Timestep is frozen at 999: the sampler is one-step."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("t", torch.tensor([999.0]))

    def forward(self, sample, encoder_hidden_states):
        return self.m(sample, self.t, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]


class DecoderWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.d = m.decoder

    def forward(self, latents):
        return self.d(latents)


def load_torch(sdxs: str, taesd: str):
    from diffusers import AutoencoderTiny, UNet2DConditionModel
    from diffusers.models.attention_processor import AttnProcessor
    from transformers import CLIPTextModel

    text = CLIPTextModel.from_pretrained(
        sdxs, subfolder="text_encoder", use_safetensors=True, attn_implementation="eager", torch_dtype=torch.float32
    ).eval()
    unet = UNet2DConditionModel.from_pretrained(
        sdxs, subfolder="unet", use_safetensors=True, torch_dtype=torch.float32
    ).eval()
    unet.set_attn_processor(AttnProcessor())
    dec = AutoencoderTiny.from_pretrained(taesd, use_safetensors=True, torch_dtype=torch.float32).eval()
    return TextWrap(text).eval(), UNetWrap(unet).eval(), DecoderWrap(dec).eval()


def export_tokenizer(sdxs: str, out: Path) -> dict:
    """Runtime uses the light `tokenizers` package, not transformers."""
    from transformers import CLIPTokenizerFast

    tok = CLIPTokenizerFast.from_pretrained(sdxs, subfolder="tokenizer")
    d = out / "tokenizer"
    d.mkdir(parents=True, exist_ok=True)
    tok.backend_tokenizer.save(str(d / "tokenizer.json"))
    meta = {
        "bos": tok.bos_token_id,
        "eos": tok.eos_token_id,
        "pad": tok.pad_token_id,
        "max_length": tok.model_max_length,
    }
    (d / "meta.json").write_text(json.dumps(meta))
    return meta


def fetch(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print("have", dst.name)
        return
    print("fetch", url)
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)


def fetch_verifier(out: Path) -> None:
    fetch(YOLOX_URL, out / "yolox.onnx")
    for name, path in CLIP_FILES.items():
        fetch(f"{CLIP_REPO}/{path}", out / name)


def export_onnx(module, example, out: Path, name: str, inputs: list[str], axes: dict) -> None:
    """Weights go in a separate .data file, which ONNX Runtime memory-maps instead of copying."""
    import onnx

    tmp = out / f"{name}.tmp.onnx"
    with torch.no_grad():
        torch.onnx.export(
            module, example, str(tmp), dynamo=False, opset_version=17,
            input_names=inputs, output_names=["output"], dynamic_axes=axes,
        )
    proto = onnx.load(str(tmp), load_external_data=True)
    onnx.save_model(
        proto, str(out / f"{name}.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
        location=f"{name}.onnx.data", size_threshold=1024,
    )
    tmp.unlink()
    print(f"exported {name} ONNX {(out / f'{name}.onnx.data').stat().st_size / 1e6:.0f} MB", flush=True)


def export(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text, unet, dec = load_torch(args.sdxs, args.taesd)
    spatial = {2: "h", 3: "w"}

    export_onnx(text, (torch.zeros((1, 77), dtype=torch.long),), out, "text_encoder", ["input_ids"], {})
    export_onnx(
        unet, (torch.zeros(1, 4, 64, 64), torch.zeros(1, 77, 768)), out, "unet",
        ["sample", "encoder_hidden_states"], {"sample": spatial, "output": spatial},
    )
    # Small enough to keep inline.
    with torch.no_grad():
        torch.onnx.export(
            dec, (torch.zeros(1, 4, 64, 64),), str(out / "taesd_decoder.onnx"), dynamo=False, opset_version=17,
            input_names=["latents"], output_names=["output"], dynamic_axes={"latents": spatial, "output": spatial},
        )

    # Optional OpenVINO UNet, FP16, for KSHANA_UNET_BACKEND=openvino.
    t0 = time.time()
    with torch.no_grad():
        m = ov.convert_model(
            unet,
            example_input=(torch.zeros(1, 4, 64, 64), torch.zeros(1, 77, 768)),
            input=[
                ("sample", ov.PartialShape([1, 4, -1, -1]), ov.Type.f32),
                ("encoder_hidden_states", ov.PartialShape([1, 77, 768]), ov.Type.f32),
            ],
        )
    for o in m.outputs:
        o.get_tensor().set_names({"output"})
    ov.save_model(m, out / "unet.xml", compress_to_fp16=True)
    print(f"exported unet IR {(out / 'unet.bin').stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f}s", flush=True)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0**2 / mse)


def to_pixels(y: np.ndarray) -> np.ndarray:
    return (np.clip((y[0] + 1) / 2, 0, 1) * 255).round().astype(np.uint8).transpose(1, 2, 0)


def check(args) -> None:
    """PyTorch vs each UNet backend: same tokens, same z. Torch is freed before any runtime loads."""
    from transformers import CLIPTokenizer

    from kshana.generator import ONNXRUNTIME, OPENVINO, Generator

    tok = CLIPTokenizer.from_pretrained(args.sdxs, subfolder="tokenizer")
    text, unet, dec = load_torch(args.sdxs, args.taesd)
    refs = []
    for prompt, seed in zip(PROMPTS, SEEDS):
        ids = tok(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np")["input_ids"]
        z = Generator.noise(seed, 64)
        with torch.no_grad():
            emb_t = text(torch.from_numpy(ids.astype(np.int64)))
            eps_t = unet(torch.from_numpy(z), emb_t)
            x0_t = (torch.from_numpy(z) - SQRT_ONE_MINUS_ALPHA * eps_t) / SQRT_ALPHA
            refs.append((prompt, seed, ids, to_pixels(dec(x0_t).numpy())))
    del text, unet, dec
    gc.collect()

    for backend in args.backends or (ONNXRUNTIME, OPENVINO):
        gen = Generator(Path(args.out), threads=args.threads, unet_backend=backend)
        results = []
        for prompt, seed, ids, img_t in refs:
            assert (gen.tokenize(prompt) == ids).all(), "runtime tokenizer differs from transformers"
            img = gen.generate_from_embedding(gen.encode(prompt), seed=seed, latent=64)
            results.append(round(psnr(img_t, img), 2))
        print("RESULT " + json.dumps({"unet_backend": backend, "psnr_db": results, "mean_db": round(float(np.mean(results)), 2)}), flush=True)
        del gen
        gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdxs", default="IDKiro/sdxs-512-dreamshaper")
    ap.add_argument("--taesd", default="madebyollin/taesd")
    ap.add_argument("--out", default="server/models")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--backends", nargs="*", help="UNet backends to check; default both")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--skip-verifier", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = out / ".gitignore"
    if not keep.exists():
        keep.write_text("*\n")
    if not args.skip_export:
        meta = export_tokenizer(args.sdxs, out)
        print("tokenizer", meta)
        export(args)
    if not args.skip_verifier:
        fetch_verifier(out)
    if not args.skip_check:
        check(args)
    print("disk free GB", round(shutil.disk_usage(out).free / 1e9, 1))


if __name__ == "__main__":
    main()
