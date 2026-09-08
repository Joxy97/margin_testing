"""Torch-owned coefficients paired with an immutable authoritative host QUBO."""

from dataclasses import dataclass
from typing import Any

from ...optimization_problem.qubo_problem import QUBOProblem


@dataclass(frozen=True)
class TorchQUBO:
    source: QUBOProblem
    linear: Any
    heads: Any
    tails: Any
    biases: Any

    @property
    def numericMemoryBytes(self):
        return sum(value.numel() * value.element_size()
                   for value in (self.linear, self.heads, self.tails, self.biases))

    def toDevice(self, device):
        return TorchQUBO(self.source, *(value.to(device) for value in
                                       (self.linear, self.heads, self.tails, self.biases)))
