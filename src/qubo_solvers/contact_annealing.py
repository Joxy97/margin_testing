"""ContactAnnealing for unconstrained QUBO/Ising."""
from dataclasses import dataclass
import math
import torch
from ._dynamics import IterativeSolver, SpinObjective, finite, positive, publish, spins
from .solvers import _integer


@dataclass(frozen=True)
class ContactAnnealing(IterativeSolver):
    """C14 contact-Strang with U=normalized E(q)+penalty*sum((q^2-1)^2)/4.

    Contact coordinate z is evolved explicitly; damping is positive.
    """

    max_steps: int = 500
    time_step: float = .05
    penalty: float = 1.
    damping: float = .2

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        for name in ('time_step', 'penalty', 'damping'):
            positive(name, getattr(self, name))

    def _search(self, run, record):
        obj, q = SpinObjective(run), .1*spins(run)
        p, z = torch.zeros_like(q), torch.zeros_like(run.energies)
        half = .5*self.time_step
        for k in range(self.max_steps):
            q += half*p
            z += half*.5*(p*p).sum(1)
            gradient = obj.gradient(q) + self.penalty*q*(q*q-1)
            potential = obj.energy(q)/obj.scale + .25*self.penalty*((q*q-1)**2).sum(1)
            p -= half*gradient
            z -= half*potential
            p *= math.exp(-self.damping*self.time_step)
            z *= math.exp(-self.damping*self.time_step)
            p -= half*gradient
            z -= half*potential
            q += half*p
            z += half*.5*(p*p).sum(1)
            publish(run, q)
            run.iterations += 1
            record(k+1)
        finite(q, p, z)
        return self.max_steps

