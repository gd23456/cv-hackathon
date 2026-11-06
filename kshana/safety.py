"""Safety: prompt filter, image classifier, CLIP concept checks, visible AI-generated label.

Known limits: filters can be evaded by rewording and classifiers make mistakes.
These reduce risk, they do not guarantee safety.
"""
from __future__ import annotations

import os
import re
import unicodedata

import cv2
import numpy as np

LABEL = "AI-generated"

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

# Deliberately strict: this app is shown live, so it blocks some borderline words too.
_SEXUAL = (
    r"nudes?|naked|nudity|nudist|topless|bottomless|unclothed|nsfw|porn\w*|hentai|erotic\w*|xxx|"
    r"sex|sexy|sexual\w*|genitals?|genitalia|penis(?:es)?|cocks?|dicks?|phallus|phallic|"
    r"vaginas?|vulvas?|puss(?:y|ies)|clit\w*|breasts?|boob(?:s|ies)?|tits?|titt(?:y|ies)|nipples?|areolas?|"
    r"butts?|buttocks?|ass|asses|arse|booty|anus|anal|crotch|groin|testicles?|semen|cum|orgasm\w*|masturbat\w*|"
    r"blowjobs?|handjobs?|intercourse|fetish\w*|bdsm|orgy|orgies|lingerie|thongs?|bikinis?|underwear|panties|"
    r"undress\w*|stripper|striptease|seductive|provocative|sensual|"
    # Common Hindi and Hinglish slang.
    r"nangi|nanga|chut|choot|lund|gaand|gand|chuchi|randi|bhosd\w*"
)
_SEXUAL_PHRASES = r"private parts|in the buff|no garments|doing it|adult film|exposed body|uncovered chest|no clothes|without clothes|wearing nothing|bare skin|bare chest"
# Long, unambiguous words also caught inside a single joined-up word such as "bigpenis".
_SEXUAL_INSIDE = ("penis", "vagina", "pussy", "porn", "hentai", "blowjob", "handjob", "orgasm", "boobs", "titties", "masturbat")
_FOOD_AND_OBJECTS = re.compile(r"\b(?:chicken|turkey|duck) breasts?\b|\b(?:cigarette|rifle) butts?\b")
_MINOR = r"child|children|kid|kids|minor|minors|underage|toddler|infant|baby|babies|teen|teens|teenager|boy|boys|girl|girls|schoolgirl|schoolboy|loli|shota"
_VIOLENCE = (
    r"gore|gory|beheaded|beheading|decapitated|decapitation|dismembered|dismemberment|disembowell?ed|"
    r"mutilated|mutilation|torture|tortured|massacre|corpse|corpses|lynching|lynched"
)
_SELF_HARM = r"suicide|suicidal|self[- ]harm|hanging himself|hanging herself|hanging themselves|slit wrists?"
_HATE = r"nazi|swastika|kkk|white power"

_RULES = [
    ("sexual content", re.compile(rf"\b(?:{_SEXUAL}|{_SEXUAL_PHRASES})\b")),
    ("graphic violence", re.compile(rf"\b(?:{_VIOLENCE})\b")),
    ("self-harm", re.compile(rf"\b(?:{_SELF_HARM})\b")),
    ("hate symbols", re.compile(rf"\b(?:{_HATE})\b")),
]
_MINOR_RE = re.compile(rf"\b(?:{_MINOR})\b")
_MINOR_CONTEXT = re.compile(r"\b(?:swimsuit|bedroom|bed|shirtless|bath|bathing|shower|kiss\w*)\b")


UNREADABLE = "Please describe the image in English, so the safety filter can read it."


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = text.translate(_LEET)
    text = re.sub(r"[^a-z\s]", " ", text)
    # Collapse letter-spacing, from "n u d e" and "n.u.d.e" alike.
    text = re.sub(r"\b(?:[a-z] +){2,}[a-z]\b", lambda m: m.group(0).replace(" ", ""), text)
    return re.sub(r"\s+", " ", text).strip()


def check_prompt(text: str) -> str | None:
    """Return the refusal category, or None when the prompt is allowed."""
    letters = [c for c in text if c.isalpha()]
    latin = [c for c in letters if ord(c) < 0x250]
    if not letters or len(latin) / len(letters) < 0.7:
        return UNREADABLE
    norm = _FOOD_AND_OBJECTS.sub(" ", _normalise(text))
    for category, pattern in _RULES:
        if pattern.search(norm):
            return category
    if any(part in word for word in norm.split() for part in _SEXUAL_INSIDE):
        return "sexual content"
    if _MINOR_RE.search(norm) and _MINOR_CONTEXT.search(norm):
        return "sexual content"
    return None


# Zero-shot contrast: unsafe concepts against ordinary scenes, on the shared CLIP.
UNSAFE_CONCEPTS = [
    "a nude person",
    "naked human body",
    "exposed genitals",
    "bare breasts",
    "bare buttocks",
    "explicit pornographic sexual content",
    "people having sex",
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

# Chosen in eval/redteam.py on one set of seeds, and reported on a held-out set.
NSFW_THRESHOLD = float(os.environ.get("KSHANA_NSFW_THRESHOLD", "0.5"))
NSFW_THRESHOLD_SKIN = float(os.environ.get("KSHANA_NSFW_THRESHOLD_SKIN", "0.2"))
CONCEPT_THRESHOLD = float(os.environ.get("KSHANA_CONCEPT_THRESHOLD", "0.9"))
PROMPT_CONCEPT_THRESHOLD = float(os.environ.get("KSHANA_PROMPT_CONCEPT_THRESHOLD", "0.9"))
SKIN_RATIO = 0.35


def skin_ratio(rgb: np.ndarray) -> float:
    """Classical signal: share of pixels in the usual YCrCb skin-tone range."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycrcb, np.array([40, 135, 85], np.uint8), np.array([255, 180, 135], np.uint8))
    return float(np.count_nonzero(mask)) / mask.size


class ConceptCheck:
    """Flags an image or prompt when unsafe concepts take much of the softmax mass."""

    def __init__(self, clip, threshold: float = CONCEPT_THRESHOLD):
        self.clip = clip
        self.threshold = threshold
        self.unsafe = clip.embed_text(UNSAFE_CONCEPTS)
        self.safe = clip.embed_text(SAFE_CONCEPTS)

    def score(self, emb: np.ndarray) -> float:
        sims = np.concatenate([self.unsafe @ emb, self.safe @ emb]) * 100.0
        probs = np.exp(sims - sims.max())
        probs /= probs.sum()
        return float(probs[: len(self.unsafe)].sum())

    def flagged(self, image_emb: np.ndarray) -> bool:
        return self.score(image_emb) >= self.threshold

    def prompt_flagged(self, text_emb: np.ndarray) -> bool:
        return self.score(text_emb) >= PROMPT_CONCEPT_THRESHOLD


class NsfwClassifier:
    """Marqo/nsfw-image-detection-384. Returns the probability that an image is unsafe."""

    def __init__(self, session):
        self.session = session

    def probability(self, rgb: np.ndarray) -> float:
        small = cv2.resize(rgb, (384, 384), interpolation=cv2.INTER_CUBIC)
        x = np.transpose(small.astype(np.float32), (2, 0, 1))[None]
        return float(self.session.run(None, {"image": x})[0][0][0])


class OutputGate:
    """Every candidate passes here before it is scored, published or stored.

    It fails closed: if any check can't run, the image counts as unsafe.
    """

    def __init__(self, classifier: NsfwClassifier | None, concepts: ConceptCheck | None):
        self.classifier = classifier
        self.concepts = concepts

    def allows(self, rgb: np.ndarray, image_emb: np.ndarray) -> bool:
        if self.classifier is None or self.concepts is None:
            return False
        try:
            limit = NSFW_THRESHOLD_SKIN if skin_ratio(rgb) >= SKIN_RATIO else NSFW_THRESHOLD
            if self.classifier.probability(rgb) >= limit:
                return False
            return not self.concepts.flagged(image_emb)
        except Exception:
            return False


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
