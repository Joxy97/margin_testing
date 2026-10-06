"""C08 classical spin-coherent relaxation, using ordinary real vectors."""
from dataclasses import dataclass
import torch
from ._dynamics import IterativeSolver, positive
from ._vector_dynamics import search_vectors
from .solvers import _integer


@dataclass(frozen=True)
class SpinCoherentAnnealing(IterativeSolver):
    """Projected Euler with optional classical Landau--Lifshitz precession."""
    max_steps: int = 500
    time_step: float = .05
    penalty: float = 2.
    precession: float = 0.

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        positive('time_step', self.time_step)
        positive('penalty', self.penalty, zero=True)
        positive('precession', self.precession, zero=True)

    def _gradient(self, vector, progress, objective):
        gradient = torch.zeros_like(vector)
        gradient[..., 0] = -(1-progress)
        gradient[..., 2] = (progress*objective.gradient(vector[..., 2])
                            - self.penalty*progress*vector[..., 2])
        return gradient

    def _search(self, run, record):
        return search_vectors(self, run, record, precession=self.precession)
