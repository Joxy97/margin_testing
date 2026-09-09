"""Per-problem candidate selection using authoritative source-QUBO energy."""

from collections.abc import Iterable, Sequence
from typing import Any

import numpy
from scipy.sparse import coo_matrix

from ...optimization_problem.qubo_problem import QUBOProblem


class CandidateSelection:
    """Reduce candidate chunks with deterministic feasibility and tie rules."""

    def __init__(self, problem: QUBOProblem) -> None:
        self.problem = problem
        self.groups = tuple(tuple(group) for group in problem.iterOneHotGroups())
        self.feasible = False
        self.repairModel = None
        self.best: tuple[tuple[int, ...], float] | None = None

    def add(self, candidates: Iterable[tuple[Sequence[int], float]]) -> None:
        for sample, _solverEnergy in candidates:
            energy = self.problem.energy(sample)
            binary = tuple(int(value) for value in sample)
            valid = all(sum(binary[index] for index in group) == 1 for group in self.groups)
            if valid:
                if not self.feasible:
                    self.best = None
                self.feasible = True
            elif self.feasible:
                continue
            else:
                if self.repairModel is None:
                    self.repairModel = self._repairModel(self.problem)
                binary, energy = self._repairCandidate(binary, self.problem, self.groups, *self.repairModel)
            candidate = binary, energy
            if self.best is None or (energy, binary) < (self.best[1], self.best[0]):
                self.best = candidate

    def result(self) -> tuple[tuple[int, ...], float]:
        if self.best is None:
            raise ValueError("BQM solver returned no samples")
        return self.best

    @staticmethod
    def _repairModel(problem: QUBOProblem) -> tuple[Any, numpy.ndarray]:
        """Build the local-field representation used by categorical repair."""
        diagonal = problem.quadraticHeads == problem.quadraticTails
        linear = problem.linear.copy()
        if numpy.any(diagonal):
            numpy.add.at(
                linear,
                problem.quadraticHeads[diagonal],
                problem.quadraticBiases[diagonal],
            )
        heads = problem.quadraticHeads[~diagonal].astype(numpy.int64, copy=False)
        tails = problem.quadraticTails[~diagonal].astype(numpy.int64, copy=False)
        biases = problem.quadraticBiases[~diagonal]
        adjacency = coo_matrix(
            (
                numpy.concatenate((biases, biases)),
                (
                    numpy.concatenate((heads, tails)),
                    numpy.concatenate((tails, heads)),
                ),
            ),
            shape=(problem.variableCount, problem.variableCount),
        ).tocsc()
        return adjacency, linear

    @staticmethod
    def _repairCandidate(
        sample: Sequence[int],
        problem: QUBOProblem,
        groups: tuple[tuple[int, ...], ...],
        adjacency: Any,
        linear: numpy.ndarray,
    ) -> tuple[tuple[int, ...], float]:
        """Project and improve one sample by deterministic categorical descent."""
        repaired = numpy.asarray(sample, dtype=numpy.uint8).copy()
        maximum_sweeps = min(100, max(3, 2 * len(groups) + 1))
        for _sweep in range(maximum_sweeps):
            changed = False
            local_fields = linear + adjacency @ repaired
            for group in groups:
                variables = numpy.asarray(group, dtype=numpy.int64)
                selected = variables[repaired[variables] == 1]
                previous = int(selected[0]) if len(selected) == 1 else None
                for variable in selected:
                    repaired[variable] = 0
                    # The repair model is canonical CSC: each column has
                    # unique row indices, including after duplicate QUBO terms
                    # are summed. Update only its nonzero local fields.
                    start, stop = adjacency.indptr[variable : variable + 2]
                    local_fields[adjacency.indices[start:stop]] -= adjacency.data[start:stop]
                costs = local_fields[variables]
                best_position = int(numpy.argmin(costs))
                chosen = int(variables[best_position])
                if previous is not None:
                    previous_position = int(
                        numpy.flatnonzero(variables == previous)[0]
                    )
                    if costs[best_position] >= costs[previous_position] - 1e-12:
                        chosen = previous
                repaired[chosen] = 1
                start, stop = adjacency.indptr[chosen : chosen + 2]
                local_fields[adjacency.indices[start:stop]] += adjacency.data[start:stop]
                changed |= previous != chosen
            if not changed:
                break
        result = tuple(int(value) for value in repaired)
        return result, problem.energy(result)
