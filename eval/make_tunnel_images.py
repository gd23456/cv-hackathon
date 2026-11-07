#!/usr/bin/env python
"""Generates the hero page's images with the shipped model. Each must pass the safety gate."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import onnxruntime as ort

from eval.generate_coreml import CoreMLGenerator
from kshana import safety

PROMPTS = [
    "a red apple to the left of a blue cup on a wooden table",
    "a lighthouse on a cliff at sunset",
    "a golden retriever puppy on green grass",
    "a bowl of ramen on a dark table",
    "a sailboat on a calm sea at dawn",
    "a yellow bicycle leaning against a brick wall",
    "a snowy mountain at sunrise",
    "a small cabin beside a lake at sunset",
    "a cat sitting on a stack of old books",
]
OUT = Path("web/tunnel")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gate = safety.NsfwClassifier(ort.InferenceSession("server/models/nsfw.onnx"))
    generator = CoreMLGenerator("sdxs-dreamshaper", "A")
    saved = []
    for index, prompt in enumerate(PROMPTS):
        for seed in range(100, 106):
            image, _ = generator.generate(prompt, seed)
            if gate.probability(image) < 0.1:
                name = f"{index}.jpg"
                small = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(OUT / name), small[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
                saved.append({"src": f"tunnel/{name}", "prompt": prompt})
                break
    (OUT / "images.json").write_text(json.dumps(saved, indent=1))
    print("saved", len(saved), "of", len(PROMPTS))


if __name__ == "__main__":
    main()
