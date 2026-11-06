#!/usr/bin/env python
"""Flickr30K as a second unseen test set, reading the Hub copy's zip and CSV directly: no loader script, no extraction.

    python eval/flickr.py --zip <images zip> --csv <test_1k_flickr.csv>

Writes data/splits/flickr_1k.jsonl (one seeded caption per official test image; git-ignored, the dataset is not
redistributed) and data/features/reference/flickr1k.npz (same preprocessing as the COCO reference).
See results/flickr30k_source.json. Then, once only, for the shipped config:

    python eval/run.py --config S1-A --split data/splits/flickr_1k.jsonl --inception --final-test
    python eval/metrics.py --split flickr_1k --configs S1-A --final-test
"""
import argparse
import csv
import json
import os
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, FEATURES  # noqa: E402
from reference import real_image  # noqa: E402

SEED = 20261107
BATCH = 64
csv.field_size_limit(10**9)


def captions_from_csv(path: Path) -> list[dict]:
    """One seeded caption per image, in filename order, with a fixed generation seed."""
    with open(path, newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["filename"])
    rng = np.random.default_rng(SEED)
    out = []
    for k, r in enumerate(rows):
        assert r["split"] == "test", r["filename"]
        options = json.loads(r["raw"])
        out.append({"id": f"flickr_{PurePosixPath(r['filename']).stem}", "prompt": options[int(rng.integers(len(options)))].strip(),
                    "seed": 50000 + k, "filename": r["filename"]})
    return out


def zip_members(archive: zipfile.ZipFile) -> dict[str, str]:
    """Basename to member name, images only."""
    return {PurePosixPath(n).name: n for n in archive.namelist() if n.lower().endswith((".jpg", ".jpeg")) and not PurePosixPath(n).name.startswith(".")}


def reference_features(zip_path: Path, rows: list[dict], out: Path) -> dict:
    import torch
    from clip_models import load_clip
    from inception import InceptionFeatures
    from torchmetrics.functional.multimodal.clip_score import _get_features

    model, processor = load_clip("l14")
    dev = next(model.parameters()).device
    inception = InceptionFeatures()
    embs, incs = [], []
    with zipfile.ZipFile(zip_path) as archive:
        members = zip_members(archive)
        with torch.no_grad():
            for start in range(0, len(rows), BATCH):
                arrays = []
                for r in rows[start : start + BATCH]:
                    with archive.open(members[r["filename"]]) as f:
                        arrays.append(real_image(f))
                e = _get_features([torch.from_numpy(a).permute(2, 0, 1) for a in arrays], "image", dev, model, processor)
                embs.append((e / e.norm(p=2, dim=-1, keepdim=True)).cpu().numpy().astype(np.float32))
                incs.append(inception(arrays))
    store = {"ids": np.array([r["id"] for r in rows]), "emb_l14": np.concatenate(embs), "inception": np.concatenate(incs)}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **store)
    return store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    args = ap.parse_args()
    os.nice(10)
    rows = captions_from_csv(args.csv)
    split_path = DATA / "splits/flickr_1k.jsonl"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    store = reference_features(args.zip, rows, FEATURES / "reference/flickr1k.npz")
    print({k: v.shape for k, v in store.items()})


if __name__ == "__main__":
    main()
