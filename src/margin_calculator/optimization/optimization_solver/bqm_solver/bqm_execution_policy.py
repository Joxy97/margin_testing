"""Explicit policies for executing streams of QUBO problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
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
        solver.beginSeries()
        try:
            for context, problem in items:
                yield context, solver.solve(problem, solverParameters)
        finally:
            solver.endSeries()


@dataclass(frozen=True)
class BatchBQMExecutionPolicy(BQMExecutionPolicy[Context]):
    """Submit per-worker bounded batches through a native batch interface."""

    batchSize: int = 4
    maxBatchBytes: int | None = None
    memoryMultiplier: float = 3.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.batchSize, bool)
            or not isinstance(self.batchSize, Integral)
            or self.batchSize <= 0
        ):
            raise ValueError("batchSize must be a positive integer")
        if self.maxBatchBytes is not None and self.maxBatchBytes <= 0:
            raise ValueError("maxBatchBytes must be positive or None")
        if self.memoryMultiplier < 1.0:
            raise ValueError("memoryMultiplier must be at least one")

    def execute(
        self,
        solver: BQMSolver,
        items: Iterable[tuple[Context, QUBOProblem]],
        solverParameters: Mapping[str, Any] | None = None,
    ) -> Iterator[tuple[Context, BQMOptimizationResult]]:
        iterator = iter(items)
        pending: tuple[Context, QUBOProblem] | None = None
        parallelism = min(solver.batchParallelism, self.batchSize)
        if parallelism <= 0:
            raise ValueError("BQM solver batchParallelism must be positive")
        effective_max_batch_bytes = (
            None
            if self.maxBatchBytes is None
            else self.maxBatchBytes * parallelism
        )
        solver.beginSeries()
        try:
            while True:
                batch: list[tuple[Context, QUBOProblem]] = []
                estimated_bytes = 0
                while len(batch) < self.batchSize:
                    try:
                        item = pending if pending is not None else next(iterator)
                    except StopIteration:
                        break
                    pending = None
                    item_bytes = int(
                        item[1].numericMemoryBytes * self.memoryMultiplier
                    )
                    if (
                        len(batch) >= parallelism
                        and effective_max_batch_bytes is not None
                        and estimated_bytes + item_bytes
                        > effective_max_batch_bytes
                    ):
                        pending = item
                        break
                    batch.append(item)
                    estimated_bytes += item_bytes
                if not batch:
                    return
                contexts, problems = zip(*batch)
                results = solver.solveMany(problems, solverParameters)
                if len(results) != len(batch):
                    raise RuntimeError("BQM solver returned an incomplete batch")
                yield from zip(contexts, results)
        finally:
            solver.endSeries()
