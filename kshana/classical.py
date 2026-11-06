"""Classical CV layer. Every op is off / always / gated so an ablation can toggle it.

Set modes in code or with KSHANA_CLASSICAL='{"clahe":"off","unsharp":"always"}'.
"gated" means the op only runs when its own measurement says the image needs it.
Assessment functions never change pixels.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

OFF, ALWAYS, GATED = "off", "always", "gated"
DEFAULT_MODES = {
    "inpaint": GATED,
    "gray_world": GATED,
    "clahe": GATED,
    "guided_filter": GATED,
    "unsharp": GATED,
}

# Gates. Starting values; the ablation decides whether they stay.
GRAY_WORLD_CAST = 0.12  # relative channel-mean spread
CLAHE_MIN_L_STD = 40.0  # LAB L std on a 0-255 scale
SHARP_LAPLACIAN = 80.0  # Laplacian variance below this counts as soft
GRID_RATIO = 6.0  # FFT energy at the 8-px period relative to its neighbours
SATURATED_BLOB_SHARE = 0.002  # share of pixels in clipped blobs
BLANK_STD = 4.0  # below this grey std the frame is black or blank

_BRISQUE_DIR = Path(__file__).parent / "assets"


def modes_from_env() -> dict[str, str]:
    modes = dict(DEFAULT_MODES)
    raw = os.environ.get("KSHANA_CLASSICAL")
    if raw:
        for name, mode in json.loads(raw).items():
            if name not in modes or mode not in {OFF, ALWAYS, GATED}:
                raise ValueError(f"bad classical setting {name}={mode}")
            modes[name] = mode
    return modes


# --- Assessment -----------------------------------------------------------------


def laplacian_variance(rgb: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def colourfulness(rgb: np.ndarray) -> float:
    """Hasler and Suesstrunk (2003)."""
    r, g, b = [c.astype(np.float32) for c in cv2.split(rgb)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def brisque(rgb: np.ndarray) -> float | None:
    """Lower is better. None when the model files are missing. Gate only, never a headline metric."""
    model, rng = _BRISQUE_DIR / "brisque_model_live.yml", _BRISQUE_DIR / "brisque_range_live.yml"
    if not (model.exists() and rng.exists()):
        return None
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return float(cv2.quality.QualityBRISQUE_compute(bgr, str(model), str(rng))[0])


def grid_ratio(rgb: np.ndarray, period: int = 8) -> float:
    """FFT energy at the 8-px period vs its neighbours. High means a checkerboard from the decoder."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray -= gray.mean()
    mag = np.abs(np.fft.rfft2(gray))
    h, w = gray.shape
    fy, fx = round(h / period), round(w / period)
    if fx < 3 or fy < 3 or fy >= h // 2 - 2 or fx >= mag.shape[1] - 2:
        return 0.0
    peak_x = mag[0:2, fx - 1 : fx + 2].max()
    peak_y = mag[fy - 1 : fy + 2, 0:2].max()
    near_x = np.median(mag[0:2, np.r_[fx - 6 : fx - 2, fx + 3 : fx + 7]]) + 1e-6
    near_y = np.median(mag[np.r_[fy - 6 : fy - 2, fy + 3 : fy + 7], 0:2]) + 1e-6
    return float(max(peak_x / near_x, peak_y / near_y))


def saturated_mask(rgb: np.ndarray) -> np.ndarray:
    """Blown-out white blobs, big enough to be a defect but too small to be sky or a wall.
    Pure black is left alone: pupils and noses clip to black, and inpainting them damages faces."""
    mask = np.all(rgb >= 250, axis=-1).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    area_min, area_max = 12, 0.02 * rgb.shape[0] * rgb.shape[1]
    for i in range(1, n):
        if area_min <= stats[i, cv2.CC_STAT_AREA] <= area_max:
            keep[labels == i] = 1
    return keep


@dataclass
class Assessment:
    laplacian: float
    colourfulness: float
    grid: float
    blob_share: float
    grey_std: float
    severe: str | None  # reason a seed should be retried

    def as_dict(self) -> dict:
        return {
            "laplacian": round(self.laplacian, 1),
            "colourfulness": round(self.colourfulness, 1),
            "grid_ratio": round(self.grid, 2),
            "blob_share": round(self.blob_share, 4),
            "severe": self.severe,
        }


def assess(rgb: np.ndarray) -> Assessment:
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    std = float(grey.std())
    severe = None
    if std < BLANK_STD:
        severe = "blank frame"
    elif float(np.mean(np.all(rgb >= 250, axis=-1) | np.all(rgb <= 5, axis=-1))) > 0.5:
        severe = "mostly clipped frame"
    return Assessment(
        laplacian=laplacian_variance(rgb),
        colourfulness=colourfulness(rgb),
        grid=grid_ratio(rgb),
        blob_share=float(saturated_mask(rgb).mean()),
        grey_std=std,
        severe=severe,
    )


# --- Correction ops -------------------------------------------------------------


def channel_cast(rgb: np.ndarray) -> float:
    means = rgb.reshape(-1, 3).mean(0)
    return float((means.max() - means.min()) / max(means.mean(), 1.0))


def gray_world(rgb: np.ndarray) -> np.ndarray:
    means = rgb.reshape(-1, 3).mean(0) + 1e-6
    gain = means.mean() / means
    return np.clip(rgb.astype(np.float32) * gain, 0, 255).round().astype(np.uint8)


def clahe(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def guided_filter(rgb: np.ndarray, radius: int = 4, eps: float = 40.0) -> np.ndarray:
    return cv2.ximgproc.guidedFilter(rgb, rgb, radius, eps)


def edge_masked_unsharp(rgb: np.ndarray, amount: float = 0.6) -> np.ndarray:
    """Sharpen near edges only, so flat regions do not gain noise."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.GaussianBlur(cv2.Canny(grey, 60, 160), (0, 0), 1.5).astype(np.float32) / 255.0
    mask = np.clip(edges * 3.0, 0, 1)[..., None]
    blurred = cv2.GaussianBlur(rgb, (0, 0), 1.2).astype(np.float32)
    sharp = rgb.astype(np.float32) + amount * (rgb.astype(np.float32) - blurred)
    return np.clip(rgb * (1 - mask) + sharp * mask, 0, 255).round().astype(np.uint8)


def inpaint_blobs(rgb: np.ndarray) -> np.ndarray:
    mask = cv2.dilate(saturated_mask(rgb), np.ones((3, 3), np.uint8))
    return cv2.inpaint(rgb, mask, 3, cv2.INPAINT_TELEA)


def resize_lanczos(rgb: np.ndarray, size: int) -> np.ndarray:
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LANCZOS4)


# --- Chain ----------------------------------------------------------------------


@dataclass
class ClassicalResult:
    image: np.ndarray
    ops: list[dict] = field(default_factory=list)
    assessment: Assessment | None = None


def _decide(mode: str, gate_open: bool, why: str) -> tuple[bool, str]:
    if mode == OFF:
        return False, "off"
    if mode == ALWAYS:
        return True, "always on"
    return gate_open, why if gate_open else f"gate closed: {why}"


def enhance(rgb: np.ndarray, names_colours: bool, modes: dict[str, str] | None = None) -> ClassicalResult:
    """Run the chain. Gray-world never runs when the prompt names colours, in any mode."""
    modes = modes or DEFAULT_MODES
    before = assess(rgb)
    out = rgb
    ops: list[dict] = []

    def step(name, fn, gate_open, why, forced_skip=None):
        nonlocal out
        t0 = time.perf_counter()
        mode = modes[name]
        if forced_skip:
            run, note = False, forced_skip
        else:
            run, note = _decide(mode, gate_open, why)
        if run:
            out = fn(out)
        ops.append({"op": name, "mode": mode, "applied": run, "note": note, "ms": round((time.perf_counter() - t0) * 1000, 1)})

    step("inpaint", inpaint_blobs, before.blob_share >= SATURATED_BLOB_SHARE, f"clipped blobs {before.blob_share:.4f}")
    cast = channel_cast(out)
    step(
        "gray_world",
        gray_world,
        cast >= GRAY_WORLD_CAST,
        f"colour cast {cast:.3f}",
        forced_skip="skipped: prompt names colours" if names_colours else None,
    )
    l_std = float(cv2.cvtColor(out, cv2.COLOR_RGB2LAB)[..., 0].std())
    step("clahe", clahe, l_std < CLAHE_MIN_L_STD, f"L std {l_std:.1f}")
    step("guided_filter", guided_filter, before.grid >= GRID_RATIO, f"8-px grid ratio {before.grid:.2f}")
    lap = laplacian_variance(out)
    step("unsharp", edge_masked_unsharp, lap < SHARP_LAPLACIAN, f"Laplacian variance {lap:.0f}")
    return ClassicalResult(out, ops, before)
