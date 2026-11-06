"""Per-image features: CLIP L/14 and B/32 score and embedding (always against the original prompt), Inception optional."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_models import load_clip, score_batch  # noqa: E402

CLIP_KEYS = ("clip_l14", "clip_b32", "emb_l14", "emb_b32")
BATCH = 32


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


class Extractor:
    def __init__(self, inception: bool = False):
        self.clip = {k: load_clip(k) for k in ("l14", "b32")}
        self.inception = None
        if inception:
            from inception import InceptionFeatures

            self.inception = InceptionFeatures()

    @property
    def keys(self) -> tuple[str, ...]:
        return CLIP_KEYS + (("inception",) if self.inception else ())

    def extract(self, rows: list[dict], img_dir: Path, keys: tuple[str, ...]) -> dict:
        arrays = [read_rgb(img_dir / f"{r['id']}.png") for r in rows]
        prompts = [r["prompt"] for r in rows]
        out = {}
        for kind, (model, processor) in self.clip.items():
            if f"clip_{kind}" not in keys:
                continue
            scores, embs = [], []
            for start in range(0, len(rows), BATCH):
                s, e = score_batch(arrays[start : start + BATCH], prompts[start : start + BATCH], model, processor)
                scores.append(s)
                embs.append(e)
            out[f"clip_{kind}"], out[f"emb_{kind}"] = np.concatenate(scores), np.concatenate(embs)
        if "inception" in keys:
            out["inception"] = self.inception(arrays)
        return out
