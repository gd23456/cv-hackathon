"""Glue: parse, plan, then run the anytime loop over generate > classical > verify > safety."""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import classical, parser, planner, safety, verifier
from .anytime import Attempt, run_anytime
from .generator import LATENT_FOR_RES, Generator
from .inputs import CleanPrompt, fit_tokens

BENCH_PROMPT = "a red apple to the left of a blue cup on a wooden table"
DEFAULT_MODELS = Path(__file__).resolve().parent.parent / "server" / "models"
JPEG_QUALITY = 90


def models_dir() -> Path:
    return Path(os.environ.get("KSHANA_MODELS", DEFAULT_MODELS))


def configured_threads() -> int | None:
    return int(os.environ.get("KSHANA_THREADS", "0")) or None


@dataclass
class Outcome:
    status: str  # done | refused | error | cancelled
    error: str | None = None
    code: str | None = None
    timings: dict = field(default_factory=dict)


def encode_jpeg(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class Pipeline:
    def __init__(self, models: Path | None = None, threads: int | None = None, classical_modes: dict | None = None):
        self.models = Path(models) if models else models_dir()
        self.threads = threads or configured_threads()
        self.classical_modes = classical_modes or classical.modes_from_env()
        self.costs: planner.CostTable | None = None

    def load(self) -> None:
        if self.threads:
            cv2.setNumThreads(self.threads)
        self.gen = Generator(self.models, self.threads)
        self.verifier = verifier.Verifier(self.models, self.threads)
        self.concepts = safety.ConceptCheck(self.verifier.clip)
        parser.parse("warm up the parser")

    def benchmark(self) -> planner.CostTable:
        """Measure this machine's costs. Doubles as warm-up: the first runs of each shape are discarded."""

        def timed(fn, repeat=2):
            for _ in range(repeat - 1):
                fn()
            t0 = time.perf_counter()
            result = fn()
            return (time.perf_counter() - t0) * 1000.0, result

        text_ms, emb = timed(lambda: self.gen.encode(BENCH_PROMPT))
        # Largest first: dynamic-shape buffers then get reused, which lowers peak memory.
        # 512 again last: switching shape costs about a second once, and the first click must not pay it.
        seed_ms = {}
        for res in (512, 384, 256, 512):
            ms, image = timed(lambda: self.gen.generate_from_embedding(emb, 1, LATENT_FOR_RES[res]), repeat=3)
            seed_ms[res] = ms
        classical_ms, _ = timed(lambda: classical.enhance(image, False, self.classical_modes))
        detect_ms, _ = timed(lambda: self.verifier.detector.detect(image))
        clip_image_ms, _ = timed(lambda: self.verifier.clip.embed_image(image))
        clip_text_ms, _ = timed(lambda: self.verifier.text_embedding(BENCH_PROMPT))
        self.costs = planner.CostTable(text_ms, seed_ms, classical_ms, detect_ms, clip_image_ms, clip_text_ms, self.threads)
        return self.costs

    def _attempt(self, seed, emb, plan, parsed, text_emb) -> Attempt:
        t0 = time.perf_counter()
        raw = self.gen.generate_from_embedding(emb, seed, LATENT_FOR_RES[plan.resolution])
        t1 = time.perf_counter()
        res = classical.enhance(raw, parsed.names_colours, self.classical_modes)
        t2 = time.perf_counter()
        image_emb = self.verifier.clip.embed_image(res.image)
        if self.concepts.flagged(image_emb):
            return Attempt(seed, None, -1.0, False, rejected="output safety check")
        if res.assessment.severe:
            return Attempt(seed, None, -1.0, False, rejected=res.assessment.severe)

        dets = self.verifier.detector.detect(res.image) if plan.checks == planner.FULL else None
        checklist = verifier.build_checklist(res.image, parsed, dets)
        alignment = float(image_emb @ text_emb) if text_emb is not None else None
        if alignment is not None:
            checklist.append(
                {
                    "kind": "alignment",
                    "label": "Overall match to prompt",
                    "status": verifier.INFO,
                    "detail": "CLIP similarity: higher is closer. Not a pass or fail.",
                    "value": round(alignment, 3),
                }
            )
        else:
            checklist.append(
                {
                    "kind": "alignment",
                    "label": "Overall match to prompt",
                    "status": verifier.NOT_VERIFIABLE,
                    "detail": "Skipped at this speed tier.",
                }
            )
        t3 = time.perf_counter()
        return Attempt(
            seed,
            res.image,
            verifier.score_candidate(checklist, alignment, None),
            True,
            payload={
                "checklist": checklist,
                "classical": res.ops,
                "quality": res.assessment.as_dict(),
                "times": {
                    "generate_ms": round((t1 - t0) * 1000, 1),
                    "classical_ms": round((t2 - t1) * 1000, 1),
                    "verify_ms": round((t3 - t2) * 1000, 1),
                },
            },
        )

    def run(
        self,
        clean: CleanPrompt,
        budget: str,
        device: planner.Device,
        remaining_s: float,
        waiting_behind: int,
        on_plan: Callable[[planner.Plan, list[str]], None],
        on_best: Callable[[dict], None],
        should_stop: Callable[[], bool],
    ) -> Outcome:
        """`remaining_s` is what is left of the job's 30 s limit."""
        start = time.monotonic()
        hard_deadline = start + remaining_s
        t0 = time.perf_counter()
        clean = fit_tokens(clean, self.gen.truncate)
        parsed = parser.parse(clean.text)
        parse_ms = (time.perf_counter() - t0) * 1000.0

        plan = planner.plan_job(self.costs, budget, device, parsed.complexity, waiting_behind, remaining_s)
        on_plan(plan, clean.notes)

        t0 = time.perf_counter()
        emb = self.gen.encode(clean.text)
        text_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        text_emb = self.verifier.text_embedding(clean.text) if plan.checks == planner.FULL else None
        clip_text_ms = (time.perf_counter() - t0) * 1000.0

        first_image_ms: list[float] = []

        def publish(attempt: Attempt, n: int) -> None:
            t = time.perf_counter()
            shown = classical.resize_lanczos(attempt.image, plan.deliver_res)
            b64 = encode_jpeg(safety.stamp_label(shown))
            encode_ms = (time.perf_counter() - t) * 1000.0
            if not first_image_ms:
                first_image_ms.append((time.monotonic() - start) * 1000.0)
            on_best(
                {
                    "image_b64": b64,
                    "mime": "image/jpeg",
                    "width": int(shown.shape[1]),
                    "height": int(shown.shape[0]),
                    "seed": attempt.seed,
                    "attempt": n,
                    "score": attempt.score,
                    "checklist": attempt.payload["checklist"],
                    "classical": attempt.payload["classical"],
                    "quality": attempt.payload["quality"],
                    "attempt_times": attempt.payload["times"],
                    "encode_ms": round(encode_ms, 1),
                }
            )

        result = run_anytime(
            lambda seed: self._attempt(seed, emb, plan, parsed, text_emb),
            deadline=start + plan.budget_s,
            hard_deadline=hard_deadline,
            predicted_ms=plan.attempt_ms,
            on_best=publish,
            should_stop=should_stop,
        )
        timings = {
            "parse_ms": round(parse_ms, 1),
            "text_ms": round(text_ms, 1),
            "clip_text_ms": round(clip_text_ms, 1),
            "first_image_ms": round(first_image_ms[0], 1) if first_image_ms else None,
            "compute_ms": round((time.monotonic() - start) * 1000.0, 1),
            "attempts": result.attempts,
            "attempt_ms": result.attempt_ms,
            "rejected": result.rejected,
        }
        if result.stopped == "cancelled":
            return Outcome("cancelled", "Cancelled: the page stopped checking on this job.", "abandoned", timings)
        if result.best is None:
            if "output safety check" in result.rejected:
                return Outcome("refused", "The result was blocked by the safety check. Try a different prompt.", "output_safety", timings)
            return Outcome("error", "Generation failed on every try. Please try again.", "generation_failed", timings)
        return Outcome("done", timings=timings)
