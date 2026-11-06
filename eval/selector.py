"""Best-of-N selector interface, and the runtime verifier behind it.

The server verifier (YOLOX + ViT-B/32 FP16 on onnxruntime) stands in for the iPhone verifier (MobileCLIP + D-FINE).
Its CLIP is ViT-B/32, so ViT-B/32 scores are never reported for selected images. ViT-L/14 and DETR-R101 are independent.
"""
import sys
from pathlib import Path
from typing import Protocol

import numpy as np

from common import ROOT

SELECTORS = ("clip", "detector", "both")


class Selector(Protocol):
    def scores(self, rgb: np.ndarray, prompt: str) -> dict[str, float]:
        """Returns one score per name in SELECTORS; higher is better."""


class RuntimeVerifierSelector:
    def __init__(self, models: Path = ROOT / "server/models", threads: int = 2):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from kshana import parser, verifier

        self.parser, self.verifier_module = parser, verifier
        self.verifier = verifier.Verifier(models, threads)
        self.text_cache: dict[str, np.ndarray] = {}

    def scores(self, rgb: np.ndarray, prompt: str) -> dict[str, float]:
        v = self.verifier_module
        if prompt not in self.text_cache:
            self.text_cache[prompt] = self.verifier.text_embedding(prompt)
        alignment = float(self.verifier.clip.embed_image(rgb) @ self.text_cache[prompt])
        checklist = v.build_checklist(rgb, self.parser.parse(prompt), self.verifier.detector.detect(rgb))
        passed = sum(1 for i in checklist if i["status"] == v.PASS)
        failed = sum(1 for i in checklist if i["status"] == v.FAIL)
        return {
            "clip": alignment,
            "detector": passed / (passed + failed) if passed + failed else 0.5,
            "both": float(v.score_candidate(checklist, alignment, None)),
        }
