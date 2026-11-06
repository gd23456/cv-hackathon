"""Generator factory. Every generator exposes generate(prompt, seed) -> (HWC uint8 image, timings in ms)."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HF, SQRT_ALPHA, SQRT_ONE_MINUS_ALPHA, latent_for_seed, pixels_from_decoder  # noqa: E402
from configs import CONFIGS  # noqa: E402

DIFFUSERS_ALLOW = [
    "unet/*.fp16.safetensors", "unet/config.json", "text_encoder/*.fp16.safetensors", "text_encoder/config.json",
    "vae/*.fp16.safetensors", "vae/config.json", "tokenizer/*", "scheduler/*", "model_index.json",
]


class DiffusersGenerator:
    """SD 1.5 / SD-Turbo. fp16 safetensors are loaded and computed in fp32 (plan: fp32 PyTorch; saves 2x disk)."""

    def __init__(self, repo: str, steps: int, cfg: float, device: str | None = None, weights_dir: Path = HF):
        import torch
        from diffusers import StableDiffusionPipeline
        from huggingface_hub import snapshot_download

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        path = snapshot_download(repo, local_dir=weights_dir / repo.split("/")[-1], token=False, allow_patterns=DIFFUSERS_ALLOW)
        t0 = time.perf_counter()
        pipe = StableDiffusionPipeline.from_pretrained(
            path, variant="fp16", use_safetensors=True, torch_dtype=torch.float32, safety_checker=None, feature_extractor=None,
            requires_safety_checker=False,
        )
        self.pipe = pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.load_s = time.perf_counter() - t0
        self.steps, self.cfg = steps, cfg

    def generate(self, prompt: str, seed: int) -> tuple[np.ndarray, dict]:
        z = latent_for_seed(seed)
        t0 = time.perf_counter()
        out = self.pipe(prompt, num_inference_steps=self.steps, guidance_scale=self.cfg, latents=z, output_type="np").images[0]
        return (out * 255).round().astype(np.uint8), {"total_ms": 1e3 * (time.perf_counter() - t0)}


class TorchSDXS:
    """SDXS fp32 on MPS with the same sampler and TAESD: the precision ablation's upper end."""

    def __init__(self, model: str, weights_dir: Path = HF):
        import torch
        from diffusers import AutoencoderTiny, UNet2DConditionModel
        from huggingface_hub import snapshot_download
        from transformers import CLIPTextModel, CLIPTokenizer

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        repo = {"sdxs-dreamshaper": "IDKiro/sdxs-512-dreamshaper", "sdxs-0.9": "IDKiro/sdxs-512-0.9"}[model]
        path = Path(snapshot_download(repo, local_dir=weights_dir / model, token=False, allow_patterns=[
            "unet/*.safetensors", "unet/config.json", "text_encoder/*.safetensors", "text_encoder/config.json", "tokenizer/*"]))
        t0 = time.perf_counter()
        self.unet = UNet2DConditionModel.from_pretrained(path / "unet", use_safetensors=True, torch_dtype=torch.float32).eval().to(self.device)
        self.text = CLIPTextModel.from_pretrained(path / "text_encoder", use_safetensors=True, torch_dtype=torch.float32).eval().to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(path / "tokenizer")
        self.vae = AutoencoderTiny.from_pretrained(weights_dir / "taesd", use_safetensors=True, torch_dtype=torch.float32).eval().to(self.device)
        self.load_s = time.perf_counter() - t0

    def generate(self, prompt: str, seed: int) -> tuple[np.ndarray, dict]:
        torch = self.torch
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = self.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(self.device)
            z = latent_for_seed(seed).to(self.device)
            eps = self.unet(z, 999, encoder_hidden_states=self.text(ids)[0]).sample
            y = self.vae.decoder((z - SQRT_ONE_MINUS_ALPHA * eps) / SQRT_ALPHA)
        return pixels_from_decoder(y[0].cpu().numpy()), {"total_ms": 1e3 * (time.perf_counter() - t0)}


class CoreMLWithVAE:
    """Same Core ML text and UNet as the base variant; only the decoder changes, for the decoder ablation."""

    def __init__(self, model: str, variant: str, weights_dir: Path = HF):
        import torch
        from diffusers import AutoencoderKL
        from generate_coreml import CoreMLGenerator
        from huggingface_hub import snapshot_download

        self.torch = torch
        self.base = CoreMLGenerator(model, variant)
        path = Path(snapshot_download("IDKiro/sdxs-512-dreamshaper", local_dir=weights_dir / model, token=False, allow_patterns=["vae_large/*"]))
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.vae = AutoencoderKL.from_pretrained(path / "vae_large", use_safetensors=True, torch_dtype=torch.float32).eval().to(self.device)
        self.load_s = self.base.load_s

    def generate(self, prompt: str, seed: int) -> tuple[np.ndarray, dict]:
        torch = self.torch
        z = latent_for_seed(seed).numpy()
        t0 = time.perf_counter()
        eps = self.base.eps(z, self.base.encode(prompt)).astype(np.float32)
        x0 = torch.from_numpy((z - SQRT_ONE_MINUS_ALPHA * eps) / SQRT_ALPHA).to(self.device)
        with torch.no_grad():
            y = self.vae.decode(x0 / self.vae.config.scaling_factor).sample
        return pixels_from_decoder(y[0].cpu().numpy()), {"total_ms": 1e3 * (time.perf_counter() - t0)}


class ClassicalGenerator:
    """Base generator plus one classical op from kshana.classical, every other op off. Spec: 'clahe=always'."""

    def __init__(self, base, spec: str):
        from common import ROOT

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from kshana import classical, parser

        self.classical, self.parser, self.base = classical, parser, base
        op, _, mode = spec.partition("=")
        self.op = op
        self.modes = {k: classical.OFF for k in classical.DEFAULT_MODES}
        if op != "none":
            self.modes[op] = mode
        self.load_s = base.load_s

    def generate(self, prompt: str, seed: int) -> tuple[np.ndarray, dict]:
        img, ms = self.base.generate(prompt, seed)
        t0 = time.perf_counter()
        res = self.classical.enhance(img, self.parser.parse(prompt).names_colours, self.modes)
        ms["classical_ms"] = 1e3 * (time.perf_counter() - t0)
        ms["total_ms"] += ms["classical_ms"]
        ms["applied"] = float(any(o["applied"] for o in res.ops))
        return res.image, ms


def make_generator(config: str):
    name, _, spec = config.partition("+")
    cfg = CONFIGS[name]
    if cfg["kind"] == "coreml":
        from generate_coreml import CoreMLGenerator

        gen = CoreMLGenerator(cfg["model"], cfg["variant"])
    elif cfg["kind"] == "coreml_vae":
        gen = CoreMLWithVAE(cfg["model"], cfg["variant"])
    elif cfg["kind"] == "torch_sdxs":
        gen = TorchSDXS(cfg["model"])
    else:
        gen = DiffusersGenerator(cfg["repo"], cfg["steps"], cfg["cfg"])
    return ClassicalGenerator(gen, spec) if spec else gen
