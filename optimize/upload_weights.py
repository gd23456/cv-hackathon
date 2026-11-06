#!/usr/bin/env python
"""Publish the exported weights to a public Hugging Face model repo.

    python optimize/upload_weights.py                 # dry run: lists what would be uploaded
    python optimize/upload_weights.py --yes           # creates the repo and uploads

Needs `pip install huggingface_hub` and a write token from `huggingface-cli login`.
Read the licence terms in CARD first: the SDXS weights carry OpenRAIL++ use restrictions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kshana.weights import DEFAULT_REPO, MANIFEST, missing

CARD = """---
license: other
license_name: mixed
tags:
- text-to-image
- onnx
---

ONNX exports used by the Kshana demo: one-step SDXS-512-DreamShaper with a TAESD decoder,
plus the models its checks use. Nothing here was retrained or fine-tuned.

| Files | Source | Licence |
|---|---|---|
| unet.onnx, unet.onnx.data, text_encoder.onnx, text_encoder.onnx.data, tokenizer/ | [IDKiro/sdxs-512-dreamshaper](https://huggingface.co/IDKiro/sdxs-512-dreamshaper) | CreativeML OpenRAIL++-M: the upstream use restrictions apply to anyone using these files |
| taesd_decoder.onnx | [madebyollin/taesd](https://huggingface.co/madebyollin/taesd) | MIT |
| yolox.onnx | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 |
| clip_vision.onnx, clip_text.onnx, clip_tokenizer.json | [Xenova/clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32) | MIT |
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--models", default="server/models")
    ap.add_argument("--yes", action="store_true", help="really create the public repo and upload")
    args = ap.parse_args()

    models = Path(args.models)
    absent = missing(models)
    if absent:
        sys.exit(f"Missing from {models}, run optimize/export_openvino.py first: {absent}")
    total = 0
    for name in MANIFEST:
        size = (models / name).stat().st_size
        total += size
        print(f"{size / 1e6:9.1f} MB  {name}")
    print(f"{total / 1e9:9.2f} GB  total, to https://huggingface.co/{args.repo} (public)")
    if not args.yes:
        print("Dry run. Nothing was uploaded. Add --yes to publish.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=args.repo)
    for name in MANIFEST:
        print("uploading", name, flush=True)
        api.upload_file(path_or_fileobj=str(models / name), path_in_repo=name, repo_id=args.repo)
    print(f"Done. Weights URL: https://huggingface.co/{args.repo}/resolve/main")


if __name__ == "__main__":
    main()
