"""Explicit policies for executing streams of QUBO problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import islice
from numbers import Integral
from typing import Any, Generic, TypeVar

from margin_calculator.optimization.optimization_problem.qubo_problem import (
    QUBOProblem,
)
from margin_calculator.optimization.optimization_result import (
    BQMOptimizationResult,
)

from .bqm_solver import BQMSolver

Context = TypeVar("Context")


class BQMExecutionPolicy(ABC, Generic[Context]):
    """Execute context-problem pairs without assuming solver thread safety."""

    @abstractmethod
    def execute(
        self,
        solver: BQMSolver,
        items: Iterable[tuple[Context, QUBOProblem]],
        solverParameters: Mapping[str, Any] | None = None,
    ) -> Iterator[tuple[Context, BQMOptimizationResult]]:
        raise NotImplementedError


class SequentialBQMExecutionPolicy(BQMExecutionPolicy[Context]):
    """Solve one problem at a time using one solver instance."""

    def execute(
        self,
        solver: BQMSolver,
        items: Iterable[tuple[Context, QUBOProblem]],
        solverParameters: Mapping[str, Any] | None = None,
    ) -> Iterator[tuple[Context, BQMOptimizationResult]]:
        for context, problem in items:
            yield context, solver.solve(problem, solverParameters)


@dataclass(frozen=True)
class BatchBQMExecutionPolicy(BQMExecutionPolicy[Context]):
    """Submit bounded batches through a solver's native batch interface."""

    batchSize: int = 4

    def __post_init__(self) -> None:
        if (
            isinstance(self.batchSize, bool)
            or not isinstance(self.batchSize, Integral)
            or self.batchSize <= 0
        ):
            raise ValueError("batchSize must be a positive integer")

    def execute(
        self,
        solver: BQMSolver,
        items: Iterable[tuple[Context, QUBOProblem]],
        solverParameters: Mapping[str, Any] | None = None,
    ) -> Iterator[tuple[Context, BQMOptimizationResult]]:
        iterator = iter(items)
        while batch := tuple(islice(iterator, self.batchSize)):
            contexts, problems = zip(*batch)
            results = solver.solveMany(problems, solverParameters)
            if len(results) != len(batch):
                raise RuntimeError("BQM solver returned an incomplete batch")
            yield from zip(contexts, results)
