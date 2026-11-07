#!/usr/bin/env python
"""Compact, readably labelled compression figures for the report, from the two original test sheets."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

TILE, GAP, HEAD = 256, 6, 64
INK = (31, 56, 100)
FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 21)


def rows_of(sheet: Image.Image, row: int, columns: list[int]) -> list[Image.Image]:
    top = 18 + 274 * row
    return [sheet.crop((256 * c, top, 256 * (c + 1), top + TILE)) for c in columns]


def build(source: str, versions: list[tuple[str, int]], out: str) -> None:
    sheet = Image.open(source).convert("RGB")
    columns = [0, 1]  # cup, portrait
    width = len(versions) * TILE + (len(versions) - 1) * GAP
    height = HEAD + len(columns) * TILE + (len(columns) - 1) * GAP
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, row) in enumerate(versions):
        x = i * (TILE + GAP)
        for line, text in enumerate(label.split("\n")):
            draw.text((x + TILE / 2, 6 + line * 26), text, font=FONT, fill=INK, anchor="ma")
        for j, tile in enumerate(rows_of(sheet, row, columns)):
            canvas.paste(tile, (x, HEAD + j * (TILE + GAP)))
    canvas.save(out)


build("docs/figures/silu_fix_sheet.png", [("PyTorch\nreference", 0), ("Stock FP16\nNeural Engine", 4), ("Fixed FP16\nNeural Engine", 1), ("Fixed 8-bit, shipped\nNeural Engine", 2), ("Fixed 6-bit\nNeural Engine", 3)], "docs/figures/neural_engine_fix.png")
build("docs/figures/compression_sheet.png", [("PyTorch\nreference", 0), ("FP16\nGPU", 1), ("8-bit palette\nGPU", 2), ("8-bit linear\nGPU", 3), ("6-bit palette\nGPU", 4)], "docs/figures/compression_effect.png")
