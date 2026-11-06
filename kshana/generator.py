"""One-step SDXS sampler on OpenVINO. No CFG and no negative prompts by design.

z ~ N(0,I); eps = UNet(z, t=999, emb); x0 = (z - 0.99766725 eps) / 0.06826489;
y = TAESD(x0) at scale 1; pixel = clip((y+1)/2, 0, 1) * 255.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import openvino as ov
from tokenizers import Tokenizer

SQRT_ONE_MINUS_ALPHA = 0.99766725
SQRT_ALPHA = 0.06826489
MAX_TOKENS = 77
LATENT_FOR_RES = {512: 64, 384: 48, 256: 32}


class Generator:
    def __init__(self, models: Path, threads: int | None = None):
        models = Path(models)
        self.core = ov.Core()
        cfg: dict = {
            # Pinned so Apple silicon does not silently run f16; Modal's x86 defaults to f32 too.
            ov.properties.hint.inference_precision: ov.Type.f32,
            ov.properties.hint.performance_mode: ov.properties.hint.PerformanceMode.LATENCY,
        }
        threads = threads or int(os.environ.get("KSHANA_THREADS", "0")) or None
        if threads:
            cfg[ov.properties.inference_num_threads] = int(threads)
        self.threads = threads
        self.text = self.core.compile_model(models / "text_encoder.xml", "CPU", cfg).create_infer_request()
        self.unet = self.core.compile_model(models / "unet.xml", "CPU", cfg).create_infer_request()
        self.dec = self.core.compile_model(models / "taesd_decoder.xml", "CPU", cfg).create_infer_request()
        self.tok = Tokenizer.from_file(str(models / "tokenizer" / "tokenizer.json"))
        meta = json.loads((models / "tokenizer" / "meta.json").read_text())
        self.bos, self.eos, self.pad = meta["bos"], meta["eos"], meta["pad"]
        self.tok.no_truncation()
        self.tok.no_padding()

    def count_tokens(self, prompt: str) -> int:
        """Tokens including BOS and EOS."""
        return len(self.tok.encode(prompt, add_special_tokens=False).ids) + 2

    def truncate(self, prompt: str) -> tuple[str, str]:
        """Split a prompt at the 75-content-token limit: (kept, dropped)."""
        enc = self.tok.encode(prompt, add_special_tokens=False)
        if len(enc.ids) <= MAX_TOKENS - 2:
            return prompt, ""
        cut = enc.offsets[MAX_TOKENS - 3][1]
        return prompt[:cut].strip(), prompt[cut:].strip()

    def tokenize(self, prompt: str) -> np.ndarray:
        ids = self.tok.encode(prompt, add_special_tokens=False).ids[: MAX_TOKENS - 2]
        ids = [self.bos, *ids, self.eos]
        ids += [self.pad] * (MAX_TOKENS - len(ids))
        return np.array([ids], dtype=np.int64)

    def encode(self, prompt: str) -> np.ndarray:
        return self.text.infer({"input_ids": self.tokenize(prompt)})[0].copy()

    @staticmethod
    def noise(seed: int, latent: int) -> np.ndarray:
        return np.random.default_rng(seed).standard_normal((1, 4, latent, latent), dtype=np.float32)

    def generate_from_embedding(self, emb: np.ndarray, seed: int, latent: int = 64) -> np.ndarray:
        """Return an HxWx3 uint8 RGB image."""
        z = self.noise(seed, latent)
        eps = self.unet.infer({"sample": z, "encoder_hidden_states": emb})[0]
        x0 = (z - SQRT_ONE_MINUS_ALPHA * eps) / SQRT_ALPHA
        y = self.dec.infer({"latents": x0})[0]
        return (np.clip((y[0] + 1) / 2, 0, 1) * 255).round().astype(np.uint8).transpose(1, 2, 0)
