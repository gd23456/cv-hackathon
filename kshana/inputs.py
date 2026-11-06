"""Input handling: empty, oversized, non-English. Runs before anything expensive."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

MAX_CHARS = 1000

# English function words. A 4+ word prompt with none of them is probably not English.
_EN_FUNCTION = set(
    "a an the of in on at by with and or to from for is are was were this that these those "
    "his her its their my your it he she they we two three four some over under near beside "
    "above below behind next left right".split()
)


class InputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class CleanPrompt:
    text: str
    notes: list[str] = field(default_factory=list)


def _non_latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    other = sum(1 for c in letters if "LATIN" not in unicodedata.name(c, ""))
    return other / len(letters)


def sanitize(prompt: str) -> CleanPrompt:
    """Return a cleaned prompt or raise InputError with a user-facing message."""
    text = "".join(c if c.isprintable() else " " for c in (prompt or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if not any(c.isalnum() for c in text):
        raise InputError("empty_prompt", "Type what you want to see, for example: a red bicycle by a lake.")
    if len(text) > MAX_CHARS:
        raise InputError("too_long", f"That prompt is {len(text)} characters. Please keep it under {MAX_CHARS}.")
    if _non_latin_share(text) > 0.3:
        raise InputError(
            "non_english",
            "This model only understands English prompts. Please write the prompt in English.",
        )
    notes = []
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) >= 4 and not _EN_FUNCTION.intersection(words):
        notes.append("This may not be English. The model only understands English, so the result may not match.")
    return CleanPrompt(text, notes)


def fit_tokens(clean: CleanPrompt, truncate) -> CleanPrompt:
    """Apply the 77-token limit. `truncate` returns (kept, dropped)."""
    kept, dropped = truncate(clean.text)
    if not dropped:
        return clean
    tail = dropped if len(dropped) <= 60 else dropped[:57] + "..."
    notes = [*clean.notes, f'Prompt is over the 77-token limit, so this part was ignored: "{tail}"']
    return CleanPrompt(kept, notes)
