"""Verifier: YOLOX on onnxruntime (presence, count, position), HSV histogram (colour),
CLIP (overall alignment). Thresholds follow GenEval: 0.3 presence, 0.9 count, 0.1 position margin.

The detector knows 80 COCO classes. Anything else reports "not_verifiable", never a pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .parser import COCO_INDEX, VERIFIABLE_RELATIONS, ObjectSpec, ParsedPrompt

PRESENCE_CONF = 0.3
COUNT_CONF = 0.9
POSITION_MARGIN = 0.1
NMS_IOU = 0.45
COLOUR_MIN_SHARE = 0.2

PASS, FAIL, NOT_VERIFIABLE, INFO = "pass", "fail", "not_verifiable", "info"


def _session(path: Path, threads: int | None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False  # lower peak memory; the UNet already takes most of the budget
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


@dataclass
class Detection:
    cls: str
    conf: float
    box: np.ndarray  # x0, y0, x1, y1 in image pixels


class Detector:
    """YOLOX ONNX as released by Megvii: BGR 0-255 input, letterboxed, raw grid outputs."""

    def __init__(self, path: Path, threads: int | None = None, size: int = 640):
        self.sess = _session(path, threads)
        self.input = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.size = int(shape[2]) if isinstance(shape[2], int) else size
        grids, strides = [], []
        for stride in (8, 16, 32):
            n = self.size // stride
            xs, ys = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), stride))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def detect(self, rgb: np.ndarray, min_conf: float = PRESENCE_CONF) -> list[Detection]:
        h, w = rgb.shape[:2]
        ratio = min(self.size / h, self.size / w)
        nh, nw = int(h * ratio), int(w * ratio)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)[..., ::-1]
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        p = self.sess.run(None, {self.input: x})[0][0]

        xy = (p[:, :2] + self.grids) * self.strides
        wh = np.exp(np.clip(p[:, 2:4], -10, 10)) * self.strides
        scores = p[:, 4:5] * p[:, 5:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= min_conf
        if not keep.any():
            return []
        xy, wh, cls, conf = xy[keep], wh[keep], cls[keep], conf[keep]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / ratio
        out: list[Detection] = []
        for c in np.unique(cls):
            idx = np.where(cls == c)[0]
            xywh = np.concatenate([boxes[idx, :2], boxes[idx, 2:] - boxes[idx, :2]], 1).tolist()
            kept = cv2.dnn.NMSBoxes(xywh, conf[idx].tolist(), float(min_conf), NMS_IOU)
            for k in np.array(kept).reshape(-1):
                i = idx[k]
                out.append(Detection(COCO_INDEX[int(c)], float(conf[i]), boxes[i]))
        return out


class Clip:
    """OpenAI CLIP ViT-B/32 (MIT), FP16 ONNX export. Never INT8: it damages embeddings."""

    _MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    _STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

    def __init__(self, models: Path, threads: int | None = None):
        self.vision = _session(models / "clip_vision.onnx", threads)
        self.text = _session(models / "clip_text.onnx", threads)
        self.tok = Tokenizer.from_file(str(models / "clip_tokenizer.json"))
        self.tok.no_padding()
        self.tok.no_truncation()
        self.text_inputs = {i.name for i in self.text.get_inputs()}

    def embed_text(self, texts: list[str]) -> np.ndarray:
        rows = []
        masks = []
        for t in texts:
            ids = self.tok.encode(t, add_special_tokens=True).ids[:77]
            if ids[-1] != 49407:
                ids[-1] = 49407
            masks.append([1] * len(ids) + [0] * (77 - len(ids)))
            rows.append(ids + [49407] * (77 - len(ids)))
        out = []
        for row, mask in zip(rows, masks):
            feed = {"input_ids": np.array([row], np.int64)}
            if "attention_mask" in self.text_inputs:
                feed["attention_mask"] = np.array([mask], np.int64)
            e = self.text.run(["text_embeds"], feed)[0][0]
            out.append(e / np.linalg.norm(e))
        return np.stack(out)

    def embed_image(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        s = 224 / min(h, w)
        im = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_CUBIC)
        y, x = (im.shape[0] - 224) // 2, (im.shape[1] - 224) // 2
        im = im[y : y + 224, x : x + 224].astype(np.float32) / 255.0
        im = ((im - self._MEAN) / self._STD).transpose(2, 0, 1)[None]
        e = self.vision.run(["image_embeds"], {"pixel_values": im})[0][0]
        return e / np.linalg.norm(e)


# --- Colour ---------------------------------------------------------------------


def colour_shares(rgb_region: np.ndarray) -> dict[str, float]:
    """Share of pixels per named colour. OpenCV hue runs 0-179."""
    hsv = cv2.cvtColor(rgb_region, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    chromatic = (s >= 60) & (v >= 50)
    masks = {
        "black": v < 50,
        "white": (s < 40) & (v >= 190),
        "gray": (s < 60) & (v >= 50) & ~((s < 40) & (v >= 190)),
        "red": chromatic & ((h <= 7) | (h >= 172)) & ~(v < 110),
        "brown": chromatic & (h <= 22) & (v < 170),
        "orange": chromatic & (h > 7) & (h <= 22) & (v >= 170),
        "yellow": chromatic & (h > 22) & (h <= 34),
        "green": chromatic & (h > 34) & (h <= 82),
        "cyan": chromatic & (h > 82) & (h <= 100),
        "blue": chromatic & (h > 100) & (h <= 130),
        "purple": chromatic & (h > 130) & (h <= 155),
        "pink": chromatic & (h > 155) & (h < 172),
    }
    total = float(h.size)
    return {name: float(m.sum()) / total for name, m in masks.items()}


def central_crop(rgb: np.ndarray, box: np.ndarray, keep: float = 0.6) -> np.ndarray:
    """Middle of the box, which reduces background bleed into the histogram."""
    x0, y0, x1, y1 = box
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * keep, (y1 - y0) * keep
    H, W = rgb.shape[:2]
    xa, xb = int(max(cx - w / 2, 0)), int(min(cx + w / 2, W))
    ya, yb = int(max(cy - h / 2, 0)), int(min(cy + h / 2, H))
    return rgb[ya:yb, xa:xb]


# --- Geometry -------------------------------------------------------------------


def relative_position(a: np.ndarray, b: np.ndarray, margin: float = POSITION_MARGIN) -> set[str]:
    """GenEval's rule: where A sits relative to B, factoring in object sizes."""
    ca, cb = (a[:2] + a[2:]) / 2, (b[:2] + b[2:]) / 2
    da, db = np.abs(a[2:] - a[:2]), np.abs(b[2:] - b[:2])
    offset = ca - cb
    revised = np.maximum(np.abs(offset) - margin * (da + db), 0) * np.sign(offset)
    if np.all(np.abs(revised) < 1e-3):
        return set()
    dx, dy = revised / max(float(np.linalg.norm(offset)), 1e-6)
    rel = set()
    if dx < -0.5:
        rel.add("left of")
    if dx > 0.5:
        rel.add("right of")
    if dy < -0.5:
        rel.add("above")
    if dy > 0.5:
        rel.add("below")
    return rel


# --- Checklist ------------------------------------------------------------------


def _item(kind: str, label: str, status: str, detail: str = "", value: float | None = None) -> dict:
    d = {"kind": kind, "label": label, "status": status, "detail": detail}
    if value is not None:
        d["value"] = round(float(value), 3)
    return d


def _name(obj: ObjectSpec) -> str:
    return obj.noun


def build_checklist(rgb: np.ndarray, parsed: ParsedPrompt, dets: list[Detection] | None) -> list[dict]:
    """`dets=None` means the detector did not run this attempt (tier skipped it)."""
    items: list[dict] = []
    by_class: dict[str, list[Detection]] = {}
    for d in dets or []:
        by_class.setdefault(d.cls, []).append(d)
    for lst in by_class.values():
        lst.sort(key=lambda d: -d.conf)

    skipped = "Skipped at this speed tier."
    unseen = "The detector only knows 80 COCO classes, so this can't be checked."

    for obj in parsed.objects:
        name = _name(obj)
        if obj.coco is None:
            items.append(_item("presence", name, NOT_VERIFIABLE, unseen))
            if obj.count is not None:
                items.append(_item("count", f"{obj.count} × {name}", NOT_VERIFIABLE, unseen))
            if obj.colour:
                items.append(_item("colour", f"{obj.colour} {name}", NOT_VERIFIABLE, "No box to check the colour inside."))
            continue
        if dets is None:
            items.append(_item("presence", name, NOT_VERIFIABLE, skipped))
            if obj.count is not None:
                items.append(_item("count", f"{obj.count} × {name}", NOT_VERIFIABLE, skipped))
            if obj.colour:
                items.append(_item("colour", f"{obj.colour} {name}", NOT_VERIFIABLE, skipped))
            continue
        found = by_class.get(obj.coco, [])
        best = found[0].conf if found else 0.0
        if found:
            items.append(_item("presence", name, PASS, f"Found at {best:.2f} confidence.", best))
        else:
            items.append(_item("presence", name, FAIL, "Not found by the detector.", 0.0))
        if obj.count is not None:
            sure = sum(1 for d in found if d.conf >= COUNT_CONF)
            ok = sure == obj.count
            items.append(
                _item(
                    "count",
                    f"{obj.count} × {name}",
                    PASS if ok else FAIL,
                    f"Counted {sure} at 0.9 confidence or higher ({len(found)} at 0.3).",
                    sure,
                )
            )
        if obj.colour:
            if not found:
                items.append(_item("colour", f"{obj.colour} {name}", NOT_VERIFIABLE, "Object not found, so no box to check."))
            else:
                crop = central_crop(rgb, found[0].box)
                if crop.size == 0:
                    items.append(_item("colour", f"{obj.colour} {name}", NOT_VERIFIABLE, "Box too small."))
                else:
                    share = colour_shares(crop).get(obj.colour, 0.0)
                    items.append(
                        _item(
                            "colour",
                            f"{obj.colour} {name}",
                            PASS if share >= COLOUR_MIN_SHARE else FAIL,
                            f"{share:.0%} of the box centre is {obj.colour}.",
                            share,
                        )
                    )

    for rel in parsed.relations:
        a, b = parsed.objects[rel.subject], parsed.objects[rel.target]
        label = f"{_name(a)} {rel.relation} {_name(b)}"
        if rel.relation not in VERIFIABLE_RELATIONS:
            items.append(_item("position", label, NOT_VERIFIABLE, "Depth and adjacency can't be checked from 2D boxes."))
            continue
        if a.coco is None or b.coco is None:
            items.append(_item("position", label, NOT_VERIFIABLE, unseen))
            continue
        if dets is None:
            items.append(_item("position", label, NOT_VERIFIABLE, skipped))
            continue
        fa, fb = by_class.get(a.coco, []), by_class.get(b.coco, [])
        if not fa or not fb:
            items.append(_item("position", label, NOT_VERIFIABLE, "Need both objects found to compare positions."))
            continue
        ok = rel.relation in relative_position(fa[0].box, fb[0].box)
        items.append(_item("position", label, PASS if ok else FAIL, "Compared box centres with a 0.1 margin."))
    return items


def score_candidate(checklist: list[dict], alignment: float | None, severe: str | None) -> float:
    """Higher is better. Hard checks dominate; CLIP breaks ties and carries unverifiable prompts."""
    if severe:
        return -1.0
    passed = sum(1 for i in checklist if i["status"] == PASS)
    failed = sum(1 for i in checklist if i["status"] == FAIL)
    frac = passed / (passed + failed) if passed + failed else 0.5
    clip_part = float(np.clip((alignment or 0.0) / 0.4, 0, 1))
    return round(0.6 * frac + 0.4 * clip_part, 4)


class Verifier:
    def __init__(self, models: Path, threads: int | None = None):
        models = Path(models)
        self.detector = Detector(models / "yolox.onnx", threads)
        self.clip = Clip(models, threads)

    def text_embedding(self, text: str) -> np.ndarray:
        return self.clip.embed_text([text])[0]

