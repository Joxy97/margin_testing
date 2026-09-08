"""Bounded accelerator scoring with deterministic host-side repair."""

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy

from ...optimization_problem.qubo_problem import QUBOProblem
from .candidate_selection import CandidateSelection

Candidate = tuple[tuple[int, ...], float]
Selector = Callable[[Iterable[tuple[Sequence[int], float]], QUBOProblem], Candidate]


class TorchCandidateAccumulator:
    """Keep one feasible winner, or the best repaired candidate so far."""

    def __init__(self, torch: Any, problem: QUBOProblem, device: Any,
                 chunkSize: int, selector: Selector, coefficients=None) -> None:
        self.torch = torch
        self.problem = problem
        self.chunkSize = chunkSize
        self.selection = CandidateSelection(problem)
        self.best: Candidate | None = None
        self.feasible = False
        if coefficients is None:
            self.linear = torch.tensor(problem.linear, dtype=torch.float64, device=device)
            self.heads = torch.tensor(problem.quadraticHeads.astype(numpy.int64), device=device)
            self.tails = torch.tensor(problem.quadraticTails.astype(numpy.int64), device=device)
            self.biases = torch.tensor(problem.quadraticBiases, dtype=torch.float64, device=device)
        else:
            self.linear = coefficients.linear
            self.heads = coefficients.heads
            self.tails = coefficients.tails
            self.biases = coefficients.biases
        self.groups = tuple(tuple(group) for group in problem.iterOneHotGroups())
        self.groupVariables = torch.tensor(
            [index for group in self.groups for index in group], dtype=torch.int64, device=device
        )
        self.groupOffsets = torch.tensor(
            numpy.cumsum([0] + [len(group) for group in self.groups]), device=device
        )
        # Retain near ties for authoritative CPU rescoring, including differing
        # GPU reduction order and cancellation in large QUBOs.
        magnitude = numpy.abs(problem.linear).sum() + numpy.abs(problem.quadraticBiases).sum()
        self.tolerance = (64 * numpy.finfo(float).eps * (1 + magnitude)
                          * (1 + math.log2(1 + problem.variableCount + problem.interactionCount)))

    def add(self, samples: Any) -> None:
        """Consume a (runs, variables) device tensor."""
        torch = self.torch
        if self.groups:
            selected = samples[:, self.groupVariables].to(torch.int64)
            prefix = torch.cat((torch.zeros((len(samples), 1), device=samples.device,
                                           dtype=torch.int64), selected.cumsum(dim=1)), dim=1)
            counts = prefix[:, self.groupOffsets[1:]] - prefix[:, self.groupOffsets[:-1]]
            valid = (counts == 1).all(dim=1)
        else:
            valid = torch.ones(len(samples), device=samples.device, dtype=torch.bool)
        has_valid = bool(valid.any())
        if has_valid:
            if not self.feasible:
                self.best = None
            self.feasible = True
            values = samples[valid].to(torch.float64)
            energies = values @ self.linear + self.problem.offset
            for start in range(0, self.problem.interactionCount, self.chunkSize):
                stop = start + self.chunkSize
                products = values[:, self.heads[start:stop]] * values[:, self.tails[start:stop]]
                energies += products @ self.biases[start:stop]
            minimum = energies.min()
            candidates = values[energies <= minimum + self.tolerance].to(torch.uint8).cpu().numpy()
            result = min(
                ((tuple(int(v) for v in sample), self.problem.energy(sample)) for sample in candidates),
                key=lambda item: (item[1], item[0]),
            )
        elif self.feasible:
            return
        else:
            # Every infeasible candidate must be repaired; a low raw energy is
            # not a bound on its energy after categorical descent.
            host = samples.cpu().numpy()
            self.selection.add((sample, 0.0) for sample in host)
            result = self.selection.result()
        if self.best is None or (result[1], result[0]) < (self.best[1], self.best[0]):
            self.best = result

    def result(self) -> Candidate:
        if self.best is None:
            raise ValueError("BQM solver returned no samples")
        return self.best
