"""Shared paths, split IO, seeded latents, disk guard. Keep this module free of heavy imports."""
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLITS = ROOT / "eval/splits"
RESULTS = ROOT / "results"
GEN = DATA / "gen"
FEATURES = DATA / "features"
HF = DATA / "hf"
COREML = DATA / "coreml"

MIN_FREE_GB = 12.0  # lead lowered the floor from 15 at 00:12 on 7 Nov while disk is tight
# One-step DDPM constants at t=999 (alpha_cumprod 0.00466).
SQRT_ONE_MINUS_ALPHA = 0.99766725
SQRT_ALPHA = 0.06826489
TEST_SPLITS = {"complex_test", "flickr_1k"}


class DiskLow(RuntimeError):
    pass


def free_gb(path: Path = ROOT) -> float:
    return shutil.disk_usage(path).free / 2**30


def check_disk(min_gb: float = MIN_FREE_GB) -> float:
    free = free_gb()
    if free < min_gb:
        raise DiskLow(f"free disk {free:.1f} GB is below the {min_gb} GB floor")
    return free


def wait_for_disk(min_gb: float = MIN_FREE_GB, poll_s: int = 30) -> float:
    """Pause while free disk is under the floor (swap can take several GB for a while), then carry on."""
    announced = False
    while free_gb() < min_gb:
        if not announced:
            print(f"paused: free disk {free_gb():.1f} GB is below the {min_gb} GB floor", flush=True)
            announced = True
        time.sleep(poll_s)
    return free_gb()


def read_split(name_or_path: str) -> list[dict]:
    path = Path(name_or_path)
    if not path.suffix:
        path = SPLITS / f"{name_or_path}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def split_name(name_or_path: str) -> str:
    return Path(name_or_path).stem


def guard_test_split(name: str, final_test: bool) -> None:
    """The test split runs once, for the final report. Tuning runs must not touch it."""
    if split_name(name) in TEST_SPLITS and not final_test:
        raise SystemExit(f"{name} is the held-out test split; pass --final-test only for the final run, never for tuning")


def latent_for_seed(seed: int):
    """z ~ N(0, I), 1x4x64x64. Same generator as the spike, shared by every config so seeds are paired."""
    import torch

    gen = torch.Generator("cpu").manual_seed(int(seed))
    return torch.randn((1, 4, 64, 64), generator=gen, dtype=torch.float32)


def done_ids(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    with open(log_path) as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def hardware_label() -> str:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        chip = platform.processor()
    return chip or platform.machine()


def pixels_from_decoder(y):
    """y: CHW in [-1, 1] numpy array. Returns HWC uint8."""
    import numpy as np

    return ((np.clip(y, -1, 1) + 1) / 2 * 255).round().astype(np.uint8).transpose(1, 2, 0)
