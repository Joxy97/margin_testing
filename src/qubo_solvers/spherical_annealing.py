"""SphericalAnnealing for unconstrained QUBO/Ising."""
from dataclasses import dataclass
import math
from ._dynamics import IterativeSolver, SpinObjective, finite, positive, publish, spins, unit
from .solvers import _integer


@dataclass(frozen=True)
class SphericalAnnealing(IterativeSolver):
    """C12 projected Euler on ||q||^2=n with quartic binary penalty."""

    max_steps: int = 500
    time_step: float = .05
    penalty: float = 2.

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        positive('time_step', self.time_step)
        positive('penalty', self.penalty, zero=True)

    def _search(self, run, record):
        obj, q = SpinObjective(run), spins(run)
        n = q.shape[1]
        for k in range(self.max_steps):
            gradient = obj.gradient(q) + self.penalty*(k+1)/max(self.max_steps, 1)*q*(q*q-1)
            tangent = gradient - q*(q*gradient).sum(-1, keepdim=True)/n
            q = unit(q-self.time_step*tangent, math.sqrt(n))
            publish(run, q)
            run.iterations += 1
            record(k+1)
        finite(q)
        return self.max_steps

