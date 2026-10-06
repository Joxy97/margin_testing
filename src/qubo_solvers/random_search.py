"""RandomSearch for unconstrained QUBO/Ising."""
from dataclasses import dataclass
from ._dynamics import IterativeSolver
from .solvers import _integer


@dataclass(frozen=True)
class RandomSearch(IterativeSolver):
    """Uniform independent assignments; iterations count additional samples."""

    max_steps: int = 0

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)

    def _search(self, run, record):
        for k in range(self.max_steps):
            run.x.bernoulli_(.5, generator=run.generator)
            if run.spin:
                run.x.mul_(2).sub_(1)
            run.refresh_field()
            run.remember()
            run.iterations += 1
            record(k+1)
        return self.max_steps

