"""Fetch the exported weights at boot, for hosts where they can't be baked in."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Callable

# Everything the ONNX pipeline needs. optimize/upload_weights.py publishes this same list.
MANIFEST = [
    "unet.onnx",
    "unet.onnx.data",
    "text_encoder.onnx",
    "text_encoder.onnx.data",
    "taesd_decoder.onnx",
    "yolox.onnx",
    "clip_vision.onnx",
    "clip_text.onnx",
    "clip_tokenizer.json",
    "tokenizer/tokenizer.json",
    "tokenizer/meta.json",
]
DEFAULT_REPO = "gdbbg/kshana-weights"
CHUNK = 1 << 20


def base_url() -> str:
    return os.environ.get("KSHANA_WEIGHTS_URL", f"https://huggingface.co/{DEFAULT_REPO}/resolve/main").rstrip("/")


def missing(models: Path) -> list[str]:
    return [name for name in MANIFEST if not (Path(models) / name).exists()]


def _download(url: str, dst: Path, on_bytes: Callable[[int], None]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as res:
        expected = res.headers.get("Content-Length")
        got = 0
        with open(tmp, "wb") as f:
            while chunk := res.read(CHUNK):
                f.write(chunk)
                got += len(chunk)
                on_bytes(len(chunk))
    if expected is not None and got != int(expected):
        tmp.unlink()
        raise IOError(f"{dst.name}: got {got} bytes, expected {expected}")
    os.replace(tmp, dst)  # a half-finished download is never mistaken for a real file


def ensure(models: Path, on_progress: Callable[[str, int], None] = lambda name, total: None) -> None:
    """Download whatever is missing. `on_progress` gets the current file and total bytes so far."""
    models = Path(models)
    total = 0
    for name in missing(models):

        def counted(n: int, name: str = name) -> None:
            nonlocal total
            total += n
            on_progress(name, total)

        _download(f"{base_url()}/{name}", models / name, counted)
