"""SphericalAnnealing for unconstrained QUBO/Ising."""
from dataclasses import dataclass
import math
from . import observation
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
            fraction = observation.schedule_fraction(None)
            # Retain the original multiplication/division order when disabled.
            penalty = (self.penalty*(k+1)/max(self.max_steps, 1)
                       if fraction is None else self.penalty*fraction)
            gradient = obj.gradient(q) + penalty*q*(q*q-1)
            tangent = gradient - q*(q*gradient).sum(-1, keepdim=True)/n
            q = unit(q-self.time_step*tangent, math.sqrt(n))
            publish(run, q)
            run.iterations += 1
            record(k+1)
        finite(q)
        return self.max_steps

