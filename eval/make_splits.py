#!/usr/bin/env python
"""Write the fixed-seed prompt splits to eval/splits/.

    python eval/make_splits.py

Inputs in data/: coco/annotations/captions_val2017.json, prompts/geneval_metadata.jsonl,
prompts/compbench_spatial_val.txt.

coco_5k: one seeded caption per val2017 image. coco_1k: a seeded subset of coco_5k.
complex_dev / complex_test: GenEval + T2I-CompBench spatial val, stratified by source and tag.
Test is held out: eval scripts refuse it without --final-test.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "eval/splits"
SPLIT_SEED = 20261106
N_DEV = 200

# T2I-CompBench spatial phrase -> GenEval relation (None: presence only, relation not checkable).
RELATIONS = {
    "on the left of": "left of",
    "on the right of": "right of",
    "on the top of": "above",
    "on the bottom of": "below",
    "on side of": None,
    "next to": None,
    "near": None,
}
PROMPT_RE = re.compile(r"^an? (.+?) (" + "|".join(RELATIONS) + r") an? (.+?)\.?$")
COCO_CLASSES = """person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|dining table|toilet|tv|laptop|mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|scissors|teddy bear|hair drier|toothbrush""".split("|")
SYNONYMS = {"sofa": "couch", "man": "person", "woman": "person", "girl": "person", "boy": "person", "child": "person",
            "plane": "airplane", "bike": "bicycle", "television": "tv", "table": "dining table", "phone": "cell phone"}


def coco_class(word: str):
    word = SYNONYMS.get(word, word)
    return word if word in COCO_CLASSES else None


def parse_compbench(prompt: str):
    """Returns GenEval-style include clauses, or None when an object is outside COCO's 80 classes."""
    m = PROMPT_RE.match(prompt.strip())
    if not m:
        return None
    a, phrase, b = coco_class(m[1]), m[2], coco_class(m[3])
    if a is None or b is None:
        return None
    subject = {"class": a, "count": 1}
    if RELATIONS[phrase]:
        subject["position"] = [RELATIONS[phrase], 0]
    return [{"class": b, "count": 1}, subject]


def complex_rows() -> list[dict]:
    rows = []
    with open(DATA / "prompts/geneval_metadata.jsonl") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            rows.append({"id": f"ge_{i:04d}", "prompt": r["prompt"], "seed": 20000 + i, "source": "geneval", "tag": r["tag"],
                         "include": r.get("include", []), "exclude": r.get("exclude", [])})
    with open(DATA / "prompts/compbench_spatial_val.txt") as f:
        lines = [x.strip() for x in f if x.strip()]
    for i, prompt in enumerate(lines):
        clauses = parse_compbench(prompt)
        rows.append({"id": f"cb_{i:04d}", "prompt": prompt, "seed": 30000 + i, "source": "compbench", "tag": "spatial",
                     "include": clauses or [], "exclude": [], "verifiable": clauses is not None})
    return rows


def split_complex(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(SPLIT_SEED)
    strata = defaultdict(list)
    for r in rows:
        strata[(r["source"], r["tag"])].append(r)
    dev, test = [], []
    for key in sorted(strata):
        group = strata[key]
        order = rng.permutation(len(group))
        n_dev = round(N_DEV * len(group) / len(rows))
        dev += [group[i] for i in order[:n_dev]]
        test += [group[i] for i in order[n_dev:]]
    return sorted(dev, key=lambda r: r["id"]), sorted(test, key=lambda r: r["id"])


def coco_rows() -> list[dict]:
    rng = np.random.default_rng(SPLIT_SEED)
    with open(DATA / "coco/annotations/captions_val2017.json") as f:
        ann = json.load(f)
    by_image = defaultdict(list)
    for a in sorted(ann["annotations"], key=lambda a: a["id"]):
        by_image[a["image_id"]].append(a["caption"].strip())
    rows = []
    for k, image_id in enumerate(sorted(by_image)):
        caps = by_image[image_id]
        rows.append({"id": f"coco_{image_id:012d}", "prompt": caps[int(rng.integers(len(caps)))],
                     "seed": 40000 + k, "image_id": image_id})
    return rows


def write(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows))


def main() -> None:
    rows = complex_rows()
    dev, test = split_complex(rows)
    assert not {r["id"] for r in dev} & {r["id"] for r in test}
    write("complex_dev", dev)
    write("complex_test", test)
    cb = [r for r in rows if r["source"] == "compbench"]
    print("compbench verifiable:", sum(r["verifiable"] for r in cb), "of", len(cb))

    coco = coco_rows()
    write("coco_5k", coco)
    order = np.random.default_rng(SPLIT_SEED + 1).permutation(len(coco))[:1000]
    write("coco_1k", [coco[i] for i in sorted(order)])
    # Small dev set for ablations and E1: 1-2k COCO is the ablation protocol, this is its smoke test.
    order = np.random.default_rng(SPLIT_SEED + 2).permutation(len(coco))[:100]
    write("coco_100", [coco[i] for i in sorted(order)])


if __name__ == "__main__":
    main()
