"""Anytime loop: try seeds until the deadline, always holding the best image so far."""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Callable


SAFETY_REJECTION = "output safety check"
MAX_SAFETY_REJECTIONS = 3


@dataclass
class Attempt:
    """One evaluated seed. `usable` is False for blank frames and safety hits."""

    seed: int
    image: object
    score: float
    usable: bool
    payload: dict = field(default_factory=dict)
    rejected: str | None = None


@dataclass
class AnytimeResult:
    best: Attempt | None
    attempts: int
    attempt_ms: list[float]
    rejected: list[str]
    stopped: str  # deadline | cancelled | hard_deadline


EXTRA_TRIES_WHEN_EMPTY = 2


def run_anytime(
    attempt_fn: Callable[[int], Attempt],
    deadline: float,
    hard_deadline: float,
    predicted_ms: float,
    on_best: Callable[[Attempt, int], None] = lambda a, n: None,
    should_stop: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.monotonic,
    first_seed: int | None = None,
) -> AnytimeResult:
    """Always runs one attempt. Later attempts start only if they should finish by `deadline`.

    If nothing usable exists yet, a couple of extra tries may run past `deadline`,
    but never past `hard_deadline`.
    """
    seed = secrets.randbits(31) if first_seed is None else first_seed
    best: Attempt | None = None
    times: list[float] = []
    rejected: list[str] = []
    extra = 0
    n = 0
    stopped = "deadline"
    while True:
        if should_stop():
            stopped = "cancelled"
            break
        estimate = (max(times[-3:]) if times else predicted_ms) / 1000.0
        finish = clock() + estimate
        if n > 0:
            if best is not None and finish > deadline:
                break
            if best is None:
                if finish > hard_deadline:
                    stopped = "hard_deadline"
                    break
                if finish > deadline:
                    if extra >= EXTRA_TRIES_WHEN_EMPTY:
                        break
                    extra += 1
        t0 = clock()
        attempt = attempt_fn(seed + n)
        times.append((clock() - t0) * 1000.0)
        n += 1
        if not attempt.usable:
            rejected.append(attempt.rejected or "rejected")
            # A prompt that keeps producing rejected images isn't worth more tries.
            if rejected.count(SAFETY_REJECTION) >= MAX_SAFETY_REJECTIONS:
                stopped = "safety"
                break
        elif best is None or attempt.score > best.score:
            best = attempt
            on_best(best, n)
    return AnytimeResult(best, n, [round(t, 1) for t in times], rejected, stopped)
