"""GreedyLocalSearch for unconstrained QUBO and Ising objectives."""
from dataclasses import dataclass
import torch
from torch import Tensor
from .problems import Problem
from .results import OptimizationResult, TerminationReason
from .solvers import _integer, _solve


@dataclass(frozen=True)
class GreedyLocalSearch:
    """Best improving flip, lowest index on ties, periodic direct field refresh."""

    max_steps: int
    refresh_interval: int = 32

    def __post_init__(self):
        _integer("max_steps", self.max_steps, 0)
        _integer("refresh_interval", self.refresh_interval, 1)

    @torch.no_grad()
    def solve(self, problem: Problem, *, restarts: int = 1,
              initial_assignments: Tensor | None = None, seed: int | None = None,
              history_interval: int | None = None, batch_size: int | None = None,
              best_only: bool = False, memory_limit_bytes: int | None = None) -> OptimizationResult:
        return _solve(self, problem, restarts, initial_assignments, seed, history_interval,
                      batch_size, best_only, memory_limit_bytes)

    def _search(self, run, record):
        step = 0
        done = torch.zeros_like(run.iterations, dtype=torch.bool)
        for step in range(1, self.max_steps + 1):
            delta, columns = run.deltas().min(dim=1)
            if ((delta >= 0) & ~done).any():
                # Confirm apparent local optima with directly recomputed fields.
                run.refresh_field()
                delta, columns = run.deltas().min(dim=1)
                done |= delta >= 0
            accepted = (delta < 0) & ~done
            if not accepted.any():
                step -= 1
                break
            run.flip_rows(columns, accepted, delta)
            run.iterations += accepted
            if step % self.refresh_interval == 0:
                run.refresh()
            record(step)
        run.refresh_field()
        local = run.deltas().min(dim=1).values >= 0
        run.reasons[local] = TerminationReason.LOCAL_OPTIMUM
        return step

