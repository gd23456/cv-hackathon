"""score_batch must equal torchmetrics' public clip_score on the same images and prompts. Needs data/hf/clip-b32.

    python -m pytest tests/eval/test_clip_matches_torchmetrics.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from clip_models import load_clip, score_batch  # noqa: E402
from common import HF  # noqa: E402


@pytest.mark.skipif(not (HF / "clip-b32/model.safetensors").exists(), reason="weights not downloaded")
def test_score_matches_public_torchmetrics():
    from torchmetrics.functional.multimodal.clip_score import _clip_score_update

    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
    prompts = ["a photo of a cat", "a red car", "noise", "a blue cup on a table"]
    model, processor = load_clip("b32", torch.device("cpu"))
    ours, emb = score_batch(images, prompts, model, processor)
    theirs, n = _clip_score_update([torch.from_numpy(a).permute(2, 0, 1) for a in images], prompts, model, processor)
    assert n == 4
    assert np.allclose(ours, theirs.detach().numpy(), atol=1e-3)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
