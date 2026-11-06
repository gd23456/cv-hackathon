"""Safety: prompt blocklist, output concept check (CLIP), visible AI-generated label.

Known limits: a blocklist can be evaded by rewording, and a CLIP concept check
misses content and can misfire. These reduce risk, they do not guarantee safety.
"""
from __future__ import annotations

import re
import unicodedata

import cv2
import numpy as np

LABEL = "AI-generated"

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

_SEXUAL = (
    r"nude|nudes|naked|nudity|topless|nsfw|porn|porno|pornographic|hentai|erotic|erotica|xxx|"
    r"sex|sexy|sexual|sexualized|genitals?|penis|vagina|breasts?|boobs?|nipples?|fetish|orgy|"
    r"lingerie|undressed|stripper"
)
_MINOR = r"child|children|kid|kids|minor|minors|underage|toddler|infant|baby|teen|teens|teenager|schoolgirl|schoolboy|loli|shota"
_VIOLENCE = (
    r"gore|gory|beheaded|beheading|decapitated|decapitation|dismembered|dismemberment|disembowell?ed|"
    r"mutilated|mutilation|torture|tortured|massacre|corpse|corpses|lynching|lynched"
)
_SELF_HARM = r"suicide|suicidal|self[- ]harm|hanging himself|hanging herself|hanging themselves|slit wrists?"
_HATE = r"nazi|swastika|kkk|white power"

_RULES = [
    ("sexual content", re.compile(rf"\b(?:{_SEXUAL})\b")),
    ("graphic violence", re.compile(rf"\b(?:{_VIOLENCE})\b")),
    ("self-harm", re.compile(rf"\b(?:{_SELF_HARM})\b")),
    ("hate symbols", re.compile(rf"\b(?:{_HATE})\b")),
]
_MINOR_RE = re.compile(rf"\b(?:{_MINOR})\b")
_MINOR_CONTEXT = re.compile(r"\b(?:bikini|swimsuit|underwear|seductive|provocative|sensual|bedroom)\b")


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = text.translate(_LEET)
    # Collapse letter-spacing like "n u d e".
    text = re.sub(r"\b(?:[a-z] ){2,}[a-z]\b", lambda m: m.group(0).replace(" ", ""), text)
    return re.sub(r"[^a-z\s-]", " ", text)


def check_prompt(text: str) -> str | None:
    """Return the refusal category, or None when the prompt is allowed."""
    norm = _normalise(text)
    for category, pattern in _RULES:
        if pattern.search(norm):
            return category
    if _MINOR_RE.search(norm) and _MINOR_CONTEXT.search(norm):
        return "sexual content"
    return None


# Zero-shot contrast: unsafe concepts against ordinary scenes, on the shared CLIP.
UNSAFE_CONCEPTS = [
    "a nude person",
    "explicit pornographic sexual content",
    "graphic gore with blood and wounds",
    "a dead body",
]
SAFE_CONCEPTS = [
    "a photo of a person wearing clothes",
    "a photo of an animal",
    "a landscape photo",
    "a photo of everyday objects",
    "a photo of food",
    "a photo of a building or a street",
    "an illustration or a painting",
    "a photo of a vehicle",
]


class ConceptCheck:
    """Flags an image when an unsafe concept takes most of the softmax mass."""

    def __init__(self, clip, threshold: float = 0.5):
        self.threshold = threshold
        self.unsafe = clip.embed_text(UNSAFE_CONCEPTS)
        self.safe = clip.embed_text(SAFE_CONCEPTS)

    def score(self, image_emb: np.ndarray) -> float:
        sims = np.concatenate([self.unsafe @ image_emb, self.safe @ image_emb]) * 100.0
        probs = np.exp(sims - sims.max())
        probs /= probs.sum()
        return float(probs[: len(self.unsafe)].sum())

    def flagged(self, image_emb: np.ndarray) -> bool:
        return self.score(image_emb) >= self.threshold


def stamp_label(rgb: np.ndarray) -> np.ndarray:
    """Burn a visible label into the bottom-right corner of a copy of the image."""
    out = rgb.copy()
    h, w = out.shape[:2]
    scale = max(w, h) / 512 * 0.5
    thick = max(1, round(scale * 2))
    (tw, th), base = cv2.getTextSize(LABEL, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = max(4, round(6 * scale))
    x0, y0 = w - tw - 2 * pad, h - th - base - 2 * pad
    overlay = out[y0:h, x0:w].astype(np.float32) * 0.35
    out[y0:h, x0:w] = overlay.astype(np.uint8)
    cv2.putText(out, LABEL, (x0 + pad, h - pad - base), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out
