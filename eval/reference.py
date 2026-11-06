#!/usr/bin/env python
"""Features of the fixed reference: all 5,000 COCO val2017 images, center-cropped and resized to 512 with Lanczos.

    python eval/reference.py [--inception]

Every FID, KID and CMMD uses this one reference. Output: data/features/reference/coco5k.npz
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, FEATURES  # noqa: E402

BATCH = 64


def real_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        side = min(im.size)
        left, top = (im.width - side) // 2, (im.height - side) // 2
        return np.array(im.crop((left, top, left + side, top + side)).resize((512, 512), Image.LANCZOS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inception", action="store_true")
    args = ap.parse_args()
    os.nice(10)
    import torch
    from clip_models import load_clip
    from torchmetrics.functional.multimodal.clip_score import _get_features

    files = sorted((DATA / "coco/val2017").glob("*.jpg"))
    assert len(files) == 5000
    path = FEATURES / "reference/coco5k.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    store = dict(np.load(path)) if path.exists() else {}
    store["ids"] = np.array([f.stem for f in files])
    want_clip = "emb_l14" not in store
    want_inception = args.inception and "inception" not in store
    model = processor = dev = inception = None
    if want_clip:
        model, processor = load_clip("l14")
        dev = next(model.parameters()).device
    if want_inception:
        from inception import InceptionFeatures

        inception = InceptionFeatures()
    embs, incs = [], []
    with torch.no_grad():
        for start in range(0, len(files), BATCH):
            arrays = [real_image(f) for f in files[start : start + BATCH]]
            if want_clip:
                e = _get_features([torch.from_numpy(a).permute(2, 0, 1) for a in arrays], "image", dev, model, processor)
                embs.append((e / e.norm(p=2, dim=-1, keepdim=True)).cpu().numpy().astype(np.float32))
            if want_inception:
                incs.append(inception(arrays))
            if start % 640 == 0:
                print(start, flush=True)
    if want_clip:
        store["emb_l14"] = np.concatenate(embs)
    if want_inception:
        store["inception"] = np.concatenate(incs)
    np.savez(path, **store)
    print({k: v.shape for k, v in store.items()})


if __name__ == "__main__":
    main()
