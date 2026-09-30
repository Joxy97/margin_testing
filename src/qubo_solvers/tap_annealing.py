"""C11 TAP/Onsager correction to classical mean-field dynamics."""
from dataclasses import dataclass
from .mean_field_annealing import MeanFieldAnnealing


@dataclass(frozen=True)
class TAPAnnealing(MeanFieldAnnealing):
    """Euler in u=atanh(m), with Onsager correction and positive temperature floor."""

    def _squared_couplings(self, objective):
        return (2*objective.j/objective.scale).square()
