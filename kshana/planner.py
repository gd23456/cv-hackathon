"""Planner: picks a tier from the measured cost table, the budget, the prompt and the load."""
from __future__ import annotations

from dataclasses import asdict, dataclass

VERIFIED, STANDARD, CONSTRAINED, SURVIVAL = "Verified", "Standard", "Constrained", "Survival"
FULL, SAFETY_ONLY = "full", "safety"

BUDGET_SECONDS = {"fast": 4.0, "balanced": 10.0, "best": 24.0}
JOB_TIMEOUT_S = 30.0
LOW_MEMORY_GB = 2.0
SAVE_DATA_RES = 384
COMPLEX_PROMPT = 0.6


@dataclass
class Device:
    memory_gb: float | None = None
    cores: int | None = None
    save_data: bool = False


@dataclass
class CostTable:
    """Milliseconds measured on this machine at boot."""

    text_ms: float
    seed_ms: dict[int, float]  # resolution -> UNet + decoder
    classical_ms: float
    detect_ms: float
    clip_image_ms: float
    clip_text_ms: float
    threads: int | None = None

    def attempt_ms(self, res: int, checks: str) -> float:
        base = self.seed_ms[res] + self.classical_ms + self.clip_image_ms
        return base + self.detect_ms if checks == FULL else base

    def setup_ms(self, checks: str) -> float:
        return self.text_ms + self.clip_text_ms if checks == FULL else self.text_ms

    def as_dict(self) -> dict:
        d = asdict(self)
        d["seed_ms"] = {str(k): round(v, 1) for k, v in self.seed_ms.items()}
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in d.items()}


@dataclass
class Plan:
    tier: str
    resolution: int
    checks: str
    budget_s: float
    expected_attempts: int
    attempt_ms: float
    deliver_res: int
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_budget(budget: str, device: Device, complexity: float) -> tuple[str, str]:
    """Turn "auto" into a concrete budget, with the reason in words."""
    if budget != "auto":
        return budget, f"You chose {budget}."
    if device.save_data:
        return "fast", "Auto chose fast because your device asked to save data."
    if device.memory_gb is not None and device.memory_gb <= LOW_MEMORY_GB:
        return "fast", f"Auto chose fast because your device reports {device.memory_gb:g} GB of memory."
    if complexity >= COMPLEX_PROMPT:
        return "best", f"Auto chose best because this prompt has several constraints (complexity {complexity:.2f})."
    return "balanced", "Auto chose balanced."


def plan_job(
    costs: CostTable,
    budget: str,
    device: Device,
    complexity: float,
    waiting_behind: int,
    remaining_s: float,
) -> Plan:
    """`waiting_behind` jobs queued after this one; `remaining_s` left on this job's own timeout."""
    concrete, why_budget = resolve_budget(budget, device, complexity)
    allowed = BUDGET_SECONDS[concrete]
    notes = [why_budget]

    if waiting_behind > 0:
        fair = JOB_TIMEOUT_S / (waiting_behind + 1)
        if fair < allowed:
            allowed = fair
            who = "1 job is" if waiting_behind == 1 else f"{waiting_behind} jobs are"
            notes.append(f"{who} waiting, so this one gets {allowed:.1f} s.")
    if remaining_s < allowed:
        allowed = max(remaining_s, 0.0)
        notes.append(f"Queue time left {allowed:.1f} s before the 30 s limit.")

    ms = allowed * 1000.0
    full_512 = costs.attempt_ms(512, FULL)
    setup = costs.setup_ms(FULL)

    if 384 in costs.seed_ms:
        low_res = 384
    else:
        low_res = 512
    fits = int((ms - setup) // full_512) if ms > setup else 0

    if fits >= 2:
        tier, res, checks = VERIFIED, 512, FULL
        head = f"{tier}: {allowed:.1f} s fits about {fits} attempts at 512 px with all checks ({full_512 / 1000:.1f} s each here)."
    elif fits == 1:
        tier, res, checks = STANDARD, 512, FULL
        head = f"{tier}: {allowed:.1f} s fits one attempt at 512 px with all checks ({full_512 / 1000:.1f} s here)."
    elif ms >= setup + costs.attempt_ms(low_res, FULL):
        tier, res, checks = CONSTRAINED, low_res, FULL
        head = (
            f"{tier}: {allowed:.1f} s is under the {(setup + full_512) / 1000:.1f} s a 512 px attempt needs here, "
            f"so resolution drops to {low_res} px."
        )
    else:
        res = min(costs.seed_ms)
        tier, checks = SURVIVAL, SAFETY_ONLY
        head = f"{tier}: only {allowed:.1f} s available, so this is one {res} px attempt with the safety check and no other checks."

    attempt = costs.attempt_ms(res, checks)
    expected = max(1, int((ms - costs.setup_ms(checks)) // attempt)) if ms > 0 else 1

    deliver = res
    if device.save_data or (device.memory_gb is not None and device.memory_gb <= LOW_MEMORY_GB):
        deliver = min(res, SAVE_DATA_RES)
        if deliver < res:
            notes.append(f"Sent at {deliver} px to save data.")

    return Plan(
        tier=tier,
        resolution=res,
        checks=checks,
        budget_s=round(allowed, 2),
        expected_attempts=expected,
        attempt_ms=round(attempt, 1),
        deliver_res=deliver,
        reason=" ".join([head, *notes]),
    )
