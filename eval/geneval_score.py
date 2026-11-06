#!/usr/bin/env python
"""GenEval-style presence, count, position and colour scoring with a detector larger than the runtime verifier.

    python eval/geneval_score.py --config S1-A --split complex_dev

Rules follow djghosh13/geneval evaluation/evaluate_images.py (MIT): confidence 0.3, counting 0.9, position margin 0.1,
at most 16 objects per class, colour by CLIP zero-shot over 10 colours. Differences from that script, so results are
NOT comparable with published GenEval numbers: DETR-R101 boxes instead of Mask2Former masks; colour uses the box crop.
Prompts whose objects fall outside COCO's 80 classes are reported as not verifiable, never as passed.

Writes per-prompt rows to data/scores/<config>/<split>.csv and per-tag means to results/complex_scores.csv.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_models import device, load_clip  # noqa: E402
from common import DATA, GEN, HF, RESULTS, guard_test_split, read_split, split_name  # noqa: E402

DETECTOR = "facebook/detr-resnet-101"
THRESHOLD = 0.3
COUNT_THRESHOLD = 0.9
POSITION_MARGIN = 0.1
MAX_OBJECTS = 16
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
# GenEval class names that differ from DETR's COCO labels.
CLASS_ALIASES = {"computer mouse": "mouse", "computer keyboard": "keyboard", "tv remote": "remote", "tv": "tv", "cell phone": "cell phone"}


def relative_position(box_a, box_b) -> set[str]:
    """Position of A relative to B, factoring in object sizes. Boxes are x1, y1, x2, y2."""
    boxes = np.array([box_a[:4], box_b[:4]], dtype=float).reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised = np.maximum(np.abs(offset) - POSITION_MARGIN * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised) < 1e-3):
        return set()
    dx, dy = revised / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


def evaluate(row: dict, detected: dict[str, list], classify_colors) -> tuple[dict, str]:
    """Returns ({check: True/False}, reason). detected: class -> boxes, most confident first."""
    checks = {"presence": True, "count": True, "position": None, "color": None}
    reasons = []
    groups = []
    for req in row["include"]:
        name = CLASS_ALIASES.get(req["class"], req["class"])
        found = detected.get(name, [])[: req["count"]]
        matched = True
        if not detected.get(name):
            checks["presence"] = False
        if len(found) < req["count"]:
            checks["count"] = False
            matched = False
            reasons.append(f"expected {name}>={req['count']}, found {len(found)}")
        else:
            if "color" in req:
                colors = classify_colors(found, name)
                ok = colors.count(req["color"]) >= req["count"]
                checks["color"] = bool(ok) if checks["color"] is None else checks["color"] and bool(ok)
                if not ok:
                    matched = False
                    reasons.append(f"expected {req['color']} {name}, found {colors}")
            if "position" in req and matched:
                relation, target = req["position"]
                if groups[target] is None:
                    checks["position"] = False
                    matched = False
                    reasons.append(f"no target for {name} {relation}")
                else:
                    ok = all(relation in relative_position(a, b) for a in found for b in groups[target])
                    checks["position"] = bool(ok) if checks["position"] is None else checks["position"] and bool(ok)
                    if not ok:
                        matched = False
                        reasons.append(f"expected {name} {relation} target")
        if "color" in req and checks["color"] is None:
            checks["color"] = False
        if "position" in req and checks["position"] is None:
            checks["position"] = False
        groups.append(found if matched else None)
    for req in row.get("exclude", []):
        name = CLASS_ALIASES.get(req["class"], req["class"])
        if len(detected.get(name, [])) >= req["count"]:
            checks["count"] = False
            reasons.append(f"expected {name}<{req['count']}, found {len(detected[name])}")
    return checks, "; ".join(reasons)


class Scorer:
    def __init__(self):
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, DetrForObjectDetection

        path = snapshot_download(DETECTOR, local_dir=HF / "detr-r101", token=False,
                                 allow_patterns=["model.safetensors", "config.json", "preprocessor_config.json"])
        self.dev = device()
        self.detector = DetrForObjectDetection.from_pretrained(path, use_safetensors=True).eval().to(self.dev)
        self.processor = AutoImageProcessor.from_pretrained(path)
        self.labels = self.detector.config.id2label
        self.clip, self.clip_processor = load_clip("l14")
        self.classifiers: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def detect(self, image: Image.Image, threshold: float) -> dict[str, list]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.dev)
        outputs = self.detector(**inputs)
        sizes = torch.tensor([image.size[::-1]], device=self.dev)
        res = self.processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=sizes)[0]
        order = res["scores"].argsort(descending=True).cpu().tolist()
        found: dict[str, list] = defaultdict(list)
        for i in order:
            name = self.labels[int(res["labels"][i])]
            if len(found[name]) < MAX_OBJECTS:
                found[name].append(res["boxes"][i].cpu().tolist())
        return dict(found)

    @torch.no_grad()
    def _color_classifier(self, name: str) -> torch.Tensor:
        if name not in self.classifiers:
            weights = []
            for color in COLORS:
                prompts = [f"a photo of a {color} {name}", f"a photo of a {color}-colored {name}", f"a photo of a {color} object"]
                inp = self.clip_processor(text=prompts, return_tensors="pt", padding=True).to(self.dev)
                emb = self.clip.get_text_features(**inp)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                mean = emb.mean(0)
                weights.append(mean / mean.norm())
            self.classifiers[name] = torch.stack(weights)
        return self.classifiers[name]

    @torch.no_grad()
    def classify_colors(self, image: Image.Image, boxes: list, name: str) -> list[str]:
        crops = [image.crop(tuple(b)) for b in boxes]
        inp = self.clip_processor(images=crops, return_tensors="pt").to(self.dev)
        emb = self.clip.get_image_features(**inp)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return [COLORS[i] for i in (emb @ self._color_classifier(name).T).argmax(1).cpu().tolist()]

    def score_image(self, path: Path, row: dict) -> dict:
        with Image.open(path) as im:
            image = im.convert("RGB")
        threshold = COUNT_THRESHOLD if row["tag"] == "counting" else THRESHOLD
        detected = self.detect(image, threshold)
        checks, reason = evaluate(row, detected, lambda boxes, name: self.classify_colors(image, boxes, name))
        passed = all(v is not False for v in checks.values())
        return {**checks, "correct": passed, "reason": reason, "detected": json.dumps({k: len(v) for k, v in detected.items()})}


def is_verifiable(row: dict) -> bool:
    return bool(row["include"]) and row.get("verifiable", True)


def score_config(config: str, split: str, images: Path | None = None) -> list[dict]:
    rows = read_split(split)
    img_dir = images or GEN / config / split_name(split)
    scorer = Scorer()
    out = []
    for i, r in enumerate(rows):
        base = {"id": r["id"], "prompt": r["prompt"], "tag": r["tag"], "source": r["source"]}
        if not is_verifiable(r):
            out.append({**base, "correct": "not verifiable"})
        else:
            out.append({**base, **scorer.score_image(img_dir / f"{r['id']}.png", r)})
        if i % 50 == 0:
            print(i, flush=True)
    return out


def tag_means(rows: list[dict]) -> dict[str, tuple[float, int]]:
    by_tag = defaultdict(list)
    for r in rows:
        if r["correct"] != "not verifiable":
            by_tag[r["tag"]].append(bool(r["correct"]))
    return {t: (float(np.mean(v)), len(v)) for t, v in sorted(by_tag.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()
    guard_test_split(args.split, args.final_test)
    os.nice(10)
    from metrics import upsert_csv

    rows = score_config(args.config, args.split)
    out = DATA / "scores" / args.config / f"{split_name(args.split)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    means = tag_means(rows)
    n_not = sum(r["correct"] == "not verifiable" for r in rows)
    table = [{"config": args.config, "split": split_name(args.split), "tag": t, "accuracy": round(m, 4), "n": n, "not_verifiable_in_split": n_not,
              "detector": DETECTOR, "protocol": "GenEval rules, DETR-R101 boxes, not comparable to published GenEval"} for t, (m, n) in means.items()]
    upsert_csv(RESULTS / "complex_scores.csv", table, ("config", "split", "tag"))
    for t in table:
        print(t["tag"], t["accuracy"], t["n"])
    print("not verifiable:", n_not)


if __name__ == "__main__":
    main()
