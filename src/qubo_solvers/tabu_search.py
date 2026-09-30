"""TabuSearch for unconstrained QUBO/Ising."""
from dataclasses import dataclass
import torch
from ._dynamics import IterativeSolver
from .solvers import _integer


@dataclass(frozen=True)
class TabuSearch(IterativeSolver):
    """Best admissible flip, aspiration for a new best; deterministic ties.

    Iterations count accepted flips. If every move is tabu, release the move
    with the earliest expiry (lowest variable index on ties).
    """

    max_steps: int = 500
    tenure: int = 7
    refresh_interval: int = 32

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        _integer('tenure', self.tenure, 1)
        _integer('refresh_interval', self.refresh_interval, 1)

    def _search(self, run, record):
        expiry = torch.zeros_like(run.x, dtype=torch.int64)
        accepted = torch.ones_like(run.iterations, dtype=torch.bool)
        for k in range(self.max_steps):
            deltas = run.deltas()
            permitted = (expiry <= k) | (run.energies[:, None]+deltas < run.best[:, None])
            columns = deltas.masked_fill(~permitted, torch.inf).argmin(1)
            columns = torch.where(permitted.any(1), columns, expiry.argmin(1))
            delta = deltas[run.rows, columns]
            run.flip_rows(columns, accepted, delta)
            expiry[run.rows, columns] = k+self.tenure+1
            run.iterations += 1
            if (k+1) % self.refresh_interval == 0:
                run.refresh()
            record(k+1)
        return self.max_steps

