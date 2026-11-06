#!/usr/bin/env python
"""Side-by-side sheets of the kept sample images, for humans to judge. Same rows for every config, nothing hand-picked.

    python eval/sheet.py --split coco_1k --configs S1-A S1-FP16 S0.9-FP16 --out results/bakeoff
"""
import argparse
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GEN, ROOT, read_split, split_name  # noqa: E402

CELL = 384
PER_SHEET = 10
CAPTION_H = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "results/bakeoff")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = read_split(args.split)[: args.n]
    name = split_name(args.split)
    font = ImageFont.load_default(size=16)
    for page in range(0, len(rows), PER_SHEET):
        chunk = rows[page : page + PER_SHEET]
        sheet = Image.new("RGB", (CELL * len(args.configs), 32 + (CELL + CAPTION_H) * len(chunk)), "white")
        draw = ImageDraw.Draw(sheet)
        for c, config in enumerate(args.configs):
            draw.text((c * CELL + 8, 8), config, fill="black", font=font)
        for r_i, r in enumerate(chunk):
            y = 32 + r_i * (CELL + CAPTION_H)
            draw.text((8, y + 6), "\n".join(textwrap.wrap(r["prompt"], 110)[:2]), fill="black", font=font)
            for c, config in enumerate(args.configs):
                with Image.open(GEN / config / name / f"{r['id']}.png") as im:
                    sheet.paste(im.convert("RGB").resize((CELL, CELL), Image.LANCZOS), (c * CELL, y + CAPTION_H))
        path = args.out / f"{name}_{page // PER_SHEET:02d}.jpg"
        sheet.save(path, quality=92)
        print(path)


if __name__ == "__main__":
    main()
