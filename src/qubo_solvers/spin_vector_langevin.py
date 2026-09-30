"""SpinVectorLangevin for unconstrained QUBO/Ising."""
from dataclasses import dataclass
import math
import torch
from ._dynamics import IterativeSolver, SpinObjective, finite, normal, positive, publish, spins
from .solvers import _integer


@dataclass(frozen=True)
class SpinVectorLangevin(IterativeSolver):
    """C04 / margin_testing SVL; inertial Euler--Maruyama or stochastic Heun.

    Uses cos(theta) readout (a pi/2 rotation of the margin_testing convention).
    Heun reuses the noise increment; no general weak-order-two claim is made.
    """

    max_steps: int = 1000
    time_step: float = .02
    temperature: float = .01
    damping: float = .1
    mass: float = 1.
    integrator: str = 'heun'

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        positive('time_step', self.time_step)
        positive('mass', self.mass)
        positive('temperature', self.temperature, zero=True)
        positive('damping', self.damping, zero=True)
        if self.integrator not in ('euler', 'heun'):
            raise ValueError('integrator must be euler or heun')

    def _search(self, run, record):
        obj = SpinObjective(run)
        theta = math.pi / 2 - .01 * spins(run)
        velocity = torch.zeros_like(theta)
        dt = self.time_step
        amplitude = math.sqrt(2 * self.damping * self.temperature * dt) / self.mass

        def force(angle, momentum, fraction):
            return ((1 - fraction) * angle.cos()
                    + fraction * angle.sin() * obj.gradient(angle.cos())
                    - self.damping * momentum) / self.mass

        for k in range(self.max_steps):
            t = k / max(self.max_steps - 1, 1)
            acceleration = force(theta, velocity, t)
            noise = amplitude * normal(run, theta.shape) if amplitude else 0.
            predicted = theta + dt * velocity
            new_velocity = velocity + dt * acceleration + noise
            if self.integrator == 'heun':
                next_t = min((k + 1) / max(self.max_steps - 1, 1), 1.)
                theta = theta + .5 * dt * (velocity + new_velocity)
                velocity = velocity + .5 * dt * (
                    acceleration + force(predicted, new_velocity, next_t)) + noise
            else:
                theta, velocity = predicted, new_velocity
            theta.remainder_(2 * math.pi)
            publish(run, theta.cos())
            run.iterations += 1
            record(k + 1)
        finite(theta, velocity)
        return self.max_steps

