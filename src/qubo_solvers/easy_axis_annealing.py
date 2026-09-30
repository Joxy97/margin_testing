"""C07 easy-axis vector relaxation of an unconstrained QUBO/Ising objective."""
from dataclasses import dataclass
import torch
from ._dynamics import IterativeSolver, positive
from ._vector_dynamics import search_vectors
from .solvers import _integer


@dataclass(frozen=True)
class EasyAxisAnnealing(IterativeSolver):
    """Projected Euler on unit 3-vectors; internal easy-axis penalty ramps linearly."""
    max_steps: int = 500
    time_step: float = .05
    penalty: float = 2.

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        positive('time_step', self.time_step)
        positive('penalty', self.penalty, zero=True)

    def _gradient(self, vector, progress, objective):
        gradient = 2*torch.einsum('bjd,ji->bid', vector, objective.j)/objective.scale
        gradient[..., 2] += objective.h/objective.scale
        gradient[..., :2] += self.penalty*progress*vector[..., :2]
        return gradient

    def _search(self, run, record):
        return search_vectors(self, run, record)
