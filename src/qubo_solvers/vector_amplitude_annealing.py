"""C09 anchored vector-amplitude relaxation of unconstrained QUBO/Ising."""
from dataclasses import dataclass
import torch
from ._dynamics import IterativeSolver, positive
from ._vector_dynamics import search_vectors
from .solvers import _integer


@dataclass(frozen=True)
class VectorAmplitudeAnnealing(IterativeSolver):
    """Euclidean Euler, unit target amplitude and a fixed decoding/field anchor.

    Effective collinearity strength is penalty*progress/n to control size scaling.
    """
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
        covariance = vector.transpose(1, 2) @ vector
        covariance[:, 2, 2] += 1
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        gradient += ((vector*vector).sum(-1, keepdim=True)-1)*vector
        gradient += self.penalty*progress/vector.shape[1]*(
            trace[:, None, None]*vector-vector @ covariance)
        return gradient

    def _search(self, run, record):
        return search_vectors(self, run, record, project=False)
