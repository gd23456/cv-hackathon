"""Budget guard for the Modal driver. Pure Python so it can be tested without Modal."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Rates:
    """USD per second, read from modal.com/pricing on 6 Nov 2026."""

    gpu: float = 0.000164  # NVIDIA T4
    cpu_core: float = 0.0000131
    mem_gib: float = 0.00000222


@dataclass
class BudgetGuard:
    """Stops dispatch before the total could pass budget_usd, counting every running worker at its worst case.

    spent_usd carries over from earlier runs through a persistent ledger, so retries share one budget.
    """

    budget_usd: float
    spent_usd: float
    worst_chunk_usd: float
    startup_usd: float

    def charge(self, usd: float) -> None:
        self.spent_usd += usd

    def may_dispatch(self, in_flight: int, new_container: bool) -> bool:
        committed = self.spent_usd + (in_flight + 1) * self.worst_chunk_usd
        if new_container:
            committed += self.startup_usd
        return committed <= self.budget_usd


def worker_rate(rates: Rates, cpus: float, mem_gib: float) -> float:
    return rates.gpu + cpus * rates.cpu_core + mem_gib * rates.mem_gib
