"""MeanFieldAnnealing for unconstrained QUBO/Ising."""
from dataclasses import dataclass
from ._dynamics import IterativeSolver, SpinObjective, finite, positive, publish, spins
from .solvers import _integer


@dataclass(frozen=True)
class MeanFieldAnnealing(IterativeSolver):
    """C10 Hopfield mobility, Euler in u=atanh(m)."""

    max_steps: int = 500
    time_step: float = .1
    start_temperature: float = 1.
    end_temperature: float = .05

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        positive('time_step', self.time_step)
        positive('end_temperature', self.end_temperature)
        positive('start_temperature', self.start_temperature)
        if self.start_temperature < self.end_temperature:
            raise ValueError('start_temperature must be >= end_temperature')

    def _squared_couplings(self, objective):
        return None

    def _search(self, run, record):
        obj = SpinObjective(run)
        u = .1 * spins(run)
        squared = self._squared_couplings(obj)
        for k in range(self.max_steps):
            t = k / max(self.max_steps-1, 1)
            temperature = self.start_temperature * (self.end_temperature/self.start_temperature)**t
            m = u.tanh()
            gradient = obj.gradient(m) + temperature*u
            if squared is not None:
                gradient += m * ((1-m*m) @ squared) / temperature
            u -= self.time_step*gradient
            publish(run, u)
            run.iterations += 1
            record(k + 1)
        finite(u)
        return self.max_steps

