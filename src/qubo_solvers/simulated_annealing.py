"""SimulatedAnnealing for unconstrained QUBO and Ising objectives."""
from dataclasses import dataclass
import math
import torch
from torch import Tensor
from .problems import Problem
from .results import OptimizationResult
from .solvers import _integer, _solve


@dataclass(frozen=True)
class SimulatedAnnealing:
    """Single-variable Metropolis search; one sweep uses start_temperature."""

    sweeps: int
    start_temperature: float
    end_temperature: float

    def __post_init__(self):
        _integer("sweeps", self.sweeps, 0)
        if not (math.isfinite(self.start_temperature) and math.isfinite(self.end_temperature)
                and self.start_temperature >= self.end_temperature > 0):
            raise ValueError("temperatures must be finite and start >= end > 0")

    @torch.no_grad()
    def solve(self, problem: Problem, *, restarts: int = 1,
              initial_assignments: Tensor | None = None, seed: int | None = None,
              history_interval: int | None = None, batch_size: int | None = None,
              best_only: bool = False, memory_limit_bytes: int | None = None) -> OptimizationResult:
        return _solve(self, problem, restarts, initial_assignments, seed, history_interval,
                      batch_size, best_only, memory_limit_bytes)

    def _search(self, run, record):
        for sweep in range(self.sweeps):
            fraction = sweep / max(self.sweeps - 1, 1)
            temperature = math.exp((1 - fraction) * math.log(self.start_temperature)
                                   + fraction * math.log(self.end_temperature))
            order = torch.randperm(run.x.shape[1], generator=run.order_generator).tolist()
            for column in order:
                direction = -2 * run.x[:, column] if run.spin else 1 - 2 * run.x[:, column]
                delta = direction * run.field[:, column]
                if not run.spin:
                    delta.add_(run.q[column, column])
                uniform = torch.rand(run.x.shape[0], device=run.q.device, dtype=run.q.dtype,
                                     generator=run.generator)
                accepted = (delta <= 0) | (uniform < torch.exp(-delta.clamp_min(0) / temperature))
                run.flip_column(column, accepted, delta)
            run.iterations += 1
            run.refresh()
            record(sweep + 1)
        return self.sweeps

