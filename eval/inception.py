"""clean-fid Inception features, computed through clean-fid's own resizer and model class (mode='clean').

The model file is TorchScript, not safetensors. It is never downloaded here: eval/fetch_inception.sh
fetches it once and checks its SHA-256.
"""
from pathlib import Path

import numpy as np
import torch

WEIGHTS_DIR = Path.home() / ".cache/kshana-eval"
BATCH = 32


class InceptionFeatures:
    def __init__(self, device: str = "cpu"):
        from cleanfid.inception_torchscript import InceptionV3W
        from cleanfid.resize import build_resizer

        if not (WEIGHTS_DIR / "inception-2015-12-05.pt").exists():
            raise SystemExit("Inception weights missing; run eval/fetch_inception.sh after the exception is approved")
        self.device = torch.device(device)
        self.model = InceptionV3W(str(WEIGHTS_DIR), download=False, resize_inside=False).to(self.device).eval()
        self.resize = build_resizer("clean")  # PIL bicubic to 299x299, no quantisation

    @torch.no_grad()
    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        out = []
        for start in range(0, len(images), BATCH):
            batch = [torch.from_numpy(self.resize(a).copy()).permute(2, 0, 1) for a in images[start : start + BATCH]]
            out.append(self.model(torch.stack(batch).to(self.device)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
