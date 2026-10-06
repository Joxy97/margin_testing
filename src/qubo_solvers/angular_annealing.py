"""AngularAnnealing for unconstrained QUBO/Ising."""
from dataclasses import dataclass
import math
from . import observation
from ._dynamics import IterativeSolver, SpinObjective, finite, positive, publish, spins
from .solvers import _integer


@dataclass(frozen=True)
class AngularAnnealing(IterativeSolver):
    """C16 binary-vanishing feature flow; K=-2J/scale, overdamped Heun.

    feature_strength couples cos(theta)*sin(theta); quartic_penalty multiplies
    sin(theta)^4/4. The transverse field decays to zero, locking grows linearly.
    """

    max_steps: int = 500
    time_step: float = .1
    feature_strength: float = 1.
    locking: float = 2.
    quartic_penalty: float = 0.

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        positive('time_step', self.time_step)
        for key in ('feature_strength', 'locking', 'quartic_penalty'):
            positive(key, getattr(self, key), zero=True)

    def _rhs(self, angle, t, obj):
        x, y = angle.cos(), angle.sin()
        feature_field = -2 * ((x * y) @ obj.j) / obj.scale
        return (t * y * obj.gradient(x) + self.feature_strength * (x*x-y*y) * feature_field
                - self.locking * t * x * y - self.quartic_penalty * x * y**3
                + (1-t) * x)

    def _search(self, run, record):
        obj = SpinObjective(run)
        theta = math.pi / 2 - .2 * spins(run)
        for k in range(self.max_steps):
            t = observation.schedule_fraction(k / max(self.max_steps, 1))
            nt = observation.schedule_fraction((k + 1) / max(self.max_steps, 1))
            first = self._rhs(theta, t, obj)
            second = self._rhs(theta + self.time_step * first, nt, obj)
            theta += .5 * self.time_step * (first + second)
            theta.remainder_(2 * math.pi)
            publish(run, theta.cos())
            run.iterations += 1
            record(k + 1)
        finite(theta)
        return self.max_steps

