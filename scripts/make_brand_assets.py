#!/usr/bin/env python
"""Makes the logo and background files for the website and the iPhone app.

    node scripts/make_art.mjs phone
    python scripts/make_brand_assets.py <logo.png>

The logo image is Gemini-generated. Only the mark and the wordmark are cropped out of it, and the navy behind them is
keyed to transparent, so they sit on any background. The app icon puts the mark back on solid navy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
WEB_ART = ROOT / "web/art"
ASSETS = ROOT / "ios/App/Resources/Assets.xcassets"
MARK_BOX = (556, 142, 900, 480)
WORD_BOX = (488, 528, 920, 630)
# Colour distance from the navy: below the first it's background, above the second it's solid.
KEY = (18, 60)
ICON_MARK_WIDTH = 640
GRID_PT = 22
SCALE = 2


def key_out(crop: Image.Image, navy: np.ndarray, upscale: int) -> Image.Image:
    size = (crop.width * upscale, crop.height * upscale)
    rgb = np.asarray(crop.resize(size, Image.LANCZOS), dtype=np.float32)
    distance = np.sqrt(((rgb - navy) ** 2).sum(axis=2))
    alpha = np.clip((distance - KEY[0]) / (KEY[1] - KEY[0]), 0, 1)
    # Remove the navy that's blended into soft edges, or they'd show a dark fringe on the desk.
    colour = navy + (rgb - navy) / np.maximum(alpha[..., None], 1e-3)
    out = np.dstack([np.clip(colour, 0, 255), alpha * 255]).astype(np.uint8)
    image = Image.fromarray(out, "RGBA")
    return image.crop(image.getbbox())


def icon(mark: Image.Image, navy: np.ndarray, size: int, mark_width: int) -> Image.Image:
    tile = Image.new("RGB", (size, size), tuple(int(v) for v in navy))
    scaled = mark.resize((mark_width, round(mark.height * mark_width / mark.width)), Image.LANCZOS)
    tile.paste(scaled, ((size - scaled.width) // 2, (size - scaled.height) // 2), scaled)
    return tile


def imageset(name: str, image: Image.Image, jpeg: bool = False) -> None:
    folder = ASSETS / f"{name}.imageset"
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{name}.{'jpg' if jpeg else 'png'}"
    if jpeg:
        image.convert("RGB").save(folder / filename, quality=78, optimize=True)
    else:
        image.save(folder / filename, optimize=True)
    (folder / "Contents.json").write_text(json.dumps({
        "images": [{"filename": filename, "idiom": "universal"}],
        "info": {"author": "xcode", "version": 1},
    }))


def desk_for_phone(art: Image.Image) -> Image.Image:
    """The website's dark wash and dot grid, baked in, so the app draws one still image."""
    width, height = art.size
    wash = np.linspace(0.66, 0.76, height)[:, None, None]
    pixels = np.asarray(art.convert("RGB"), dtype=np.float32)
    pixels = pixels * (1 - wash) + np.array([8, 10, 14], dtype=np.float32) * wash
    image = Image.fromarray(pixels.astype(np.uint8), "RGB").convert("RGBA")
    dots = Image.new("RGBA", art.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(dots)
    step, radius = GRID_PT * SCALE, 1.1 * SCALE
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 255, 255, 41))
    return Image.alpha_composite(image, dots)


def main() -> None:
    source = Image.open(sys.argv[1]).convert("RGB")
    navy = np.median(np.asarray(source.crop((20, 20, 300, 300)), dtype=np.float32).reshape(-1, 3), axis=0)
    mark = key_out(source.crop(MARK_BOX), navy, 2)
    word = key_out(source.crop(WORD_BOX), navy, 2)
    print("navy", navy, "mark", mark.size, "word", word.size)

    small_mark = mark.resize((192, round(mark.height * 192 / mark.width)), Image.LANCZOS)
    small_word = word.resize((round(word.width * 96 / word.height), 96), Image.LANCZOS)
    small_mark.save(WEB_ART / "logo-mark.png", optimize=True)
    small_word.save(WEB_ART / "logo-word.png", optimize=True)
    icon(mark, navy, 64, 44).save(WEB_ART / "favicon-64.png", optimize=True)
    icon(mark, navy, 180, 118).save(WEB_ART / "apple-touch-180.png", optimize=True)

    icon(mark, navy, 1024, ICON_MARK_WIDTH).save(ASSETS / "AppIcon.appiconset/icon.png", optimize=True)
    imageset("KshanaMark", small_mark)
    imageset("KshanaWord", small_word)
    imageset("FolderBack", Image.open(WEB_ART / "back.jpg"), jpeg=True)
    phone = WEB_ART / "phone.png"
    imageset("Desk", desk_for_phone(Image.open(phone)), jpeg=True)
    phone.unlink()


if __name__ == "__main__":
    main()
