#!/usr/bin/env python
"""Saves same-prompt, same-seed images for the Compare section. Harmless prompts only,
and every image must also pass the output safety gate to be kept."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import onnxruntime as ort
import torch
from diffusers import AutoPipelineForText2Image

from eval.generate_coreml import CoreMLGenerator
from kshana import safety

PROMPTS = [
    ("apple", "a red apple to the left of a blue cup on a wooden table"),
    ("dogs", "two dogs sitting on a green sofa"),
    ("cabin", "a small cabin beside a lake at sunset, mountains in the background"),
    ("bicycle", "a bicycle next to a bench in a park"),
]
SEED = 42
OUT = Path("web/compare")
BASELINES = {
    "SD-Turbo, 1 step": ("stabilityai/sd-turbo", 1, 0.0),
    "SD 1.5, 25 steps": ("stable-diffusion-v1-5/stable-diffusion-v1-5", 25, 7.5),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gate = safety.NsfwClassifier(ort.InferenceSession("server/models/nsfw.onnx"))
    images: dict[str, list[dict]] = {key: [] for key, _ in PROMPTS}

    def keep(key: str, model: str, slug: str, rgb) -> None:
        if gate.probability(rgb) >= safety.NSFW_THRESHOLD:
            print("skipped by safety gate:", key, model)
            return
        name = f"{key}_{slug}.jpg"
        cv2.imwrite(str(OUT / name), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        images[key].append({"model": model, "src": f"compare/{name}"})

    ours = CoreMLGenerator("sdxs-dreamshaper", "A")
    for key, prompt in PROMPTS:
        image, _ = ours.generate(prompt, SEED)
        keep(key, "Kshana, 8-bit, 1 step", "kshana", image)
    del ours

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for model, (repo, steps, cfg) in BASELINES.items():
        pipe = AutoPipelineForText2Image.from_pretrained(repo, use_safetensors=True, torch_dtype=torch.float32, safety_checker=None).to(device)
        for key, prompt in PROMPTS:
            result = pipe(prompt, num_inference_steps=steps, guidance_scale=cfg, height=512, width=512,
                          generator=torch.Generator("cpu").manual_seed(SEED)).images[0]
            keep(key, model, repo.split("/")[-1], __import__("numpy").array(result))
        del pipe

    gallery = [{"prompt": prompt, "images": images[key]} for key, prompt in PROMPTS]
    (OUT / "gallery.json").write_text(json.dumps(gallery, indent=1))
    print("saved", sum(len(v) for v in images.values()))


if __name__ == "__main__":
    main()
