"""CLIP scoring via torchmetrics, loaded from safetensors only."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HF  # noqa: E402

# openai/clip-vit-base-patch32 has no safetensors on main; this commit is HF's safetensors conversion PR (refs/pr/3).
SOURCES = {
    "l14": ("openai/clip-vit-large-patch14", None),
    "b32": ("openai/clip-vit-base-patch32", "b527df4b30e5cc18bde1cc712833a741d2d8c362"),
}
ALLOW = ["model.safetensors", "*.json", "*.txt"]


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class TensorFeatures(torch.nn.Module):
    """torchmetrics expects get_*_features to return tensors; transformers 5 may return an output object."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    @staticmethod
    def _tensor(out):
        return out if torch.is_tensor(out) else out.pooler_output

    def get_image_features(self, *a, **k):
        return self._tensor(self.model.get_image_features(*a, **k))

    def get_text_features(self, *a, **k):
        return self._tensor(self.model.get_text_features(*a, **k))


def load_clip(name: str, dev: torch.device | None = None):
    from huggingface_hub import snapshot_download
    from transformers import CLIPModel, CLIPProcessor

    repo, revision = SOURCES[name]
    path = snapshot_download(repo, revision=revision, local_dir=HF / f"clip-{name}", token=False, allow_patterns=ALLOW)
    model = CLIPModel.from_pretrained(path, use_safetensors=True).eval().to(dev or device())
    return TensorFeatures(model), CLIPProcessor.from_pretrained(path)


@torch.no_grad()
def score_batch(images: list[np.ndarray], prompts: list[str], model, processor) -> tuple[np.ndarray, np.ndarray]:
    """Returns (100*cos per image, unit image embeddings), using torchmetrics' feature extraction and its score formula.

    tests/eval/test_clip_matches_torchmetrics.py checks this equals torchmetrics' public clip_score.
    """
    from torchmetrics.functional.multimodal.clip_score import _get_features

    dev = next(model.parameters()).device
    tensors = [torch.from_numpy(a).permute(2, 0, 1) for a in images]
    img = _get_features(tensors, "image", dev, model, processor)
    txt = _get_features(prompts, "text", dev, model, processor)
    img = img / img.norm(p=2, dim=-1, keepdim=True)
    txt = txt / txt.norm(p=2, dim=-1, keepdim=True)
    scores = 100 * (img * txt).sum(dim=-1)
    return scores.cpu().numpy().astype(np.float32), img.cpu().numpy().astype(np.float32)
