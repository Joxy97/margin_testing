"""Transverse-route angular flow from margin_testing."""
from dataclasses import dataclass
import math
import torch
from ._dynamics import IterativeSolver, SpinObjective, finite, positive, publish, spins
from .solvers import _integer


@dataclass(frozen=True)
class TransverseRoute(IterativeSolver):
    """Angular flow with an initial transverse phase and increasing binary locking.

    Initial angles are uniform in the hemisphere of the initial binary assignment.
    gamma couples sin(theta)^3 features. Counts are integration steps.
    """
    max_steps: int = 500
    time_step: float = .2
    feature_strength: float = 1.
    locking: float = .2
    gamma: float = 0.
    locking_start: float = -.2
    schedule_exponent: float = 1.
    integrator: str = 'heun'

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        positive('time_step', self.time_step)
        positive('feature_strength', self.feature_strength, zero=True)
        positive('locking', self.locking, zero=True)
        positive('gamma', self.gamma, zero=True)
        positive('schedule_exponent', self.schedule_exponent)
        if not math.isfinite(self.locking_start) or self.locking_start > self.locking:
            raise ValueError('locking_start must be finite and <= locking')
        if self.integrator not in ('euler', 'heun'):
            raise ValueError('integrator must be euler or heun')

    def _rhs(self, angle, progress, objective):
        x, y = angle.cos(), angle.sin()
        route = 2*((x*y) @ objective.j)/objective.scale
        drift = y*objective.gradient(x)-self.feature_strength*(x*x-y*y)*route
        if self.gamma:
            cubic_field = 2*(y**3 @ objective.j)/objective.scale
            drift -= 3*self.gamma*y*y*x*cubic_field
        locking = self.locking_start+(self.locking-self.locking_start)*progress**self.schedule_exponent
        return drift-locking*x*y

    def _search(self, run, record):
        objective = SpinObjective(run)
        angle = torch.rand(run.x.shape, device=run.q.device, dtype=run.q.dtype,
                           generator=run.generator).sub_(.5).mul_(math.pi)
        angle += (spins(run) < 0)*math.pi
        for k in range(self.max_steps):
            progress, next_progress = k/max(self.max_steps, 1), (k+1)/max(self.max_steps, 1)
            first = self._rhs(angle, progress, objective)
            if self.integrator == 'heun':
                second = self._rhs(angle+self.time_step*first, next_progress, objective)
                angle += .5*self.time_step*(first+second)
            else:
                angle += self.time_step*first
            angle.remainder_(2*math.pi)
            publish(run, angle.cos())
            run.iterations += 1
            record(k+1)
        finite(angle)
        return self.max_steps
