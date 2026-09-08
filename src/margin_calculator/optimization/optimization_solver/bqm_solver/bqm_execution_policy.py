"""Explicit policies for executing streams of QUBO problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Integral
from math import isfinite
from collections.abc import Callable

from .execution_memory import BQMExecutionMemory, ExecutionMemoryTracker
from typing import Any, Generic, TypeVar

from margin_calculator.optimization.optimization_problem.qubo_problem import (
    QUBOProblem,
)
from margin_calculator.optimization.optimization_result import (
    BQMOptimizationResult,
)

from .bqm_solver import BQMSolver
from .resource_plan import BQMResourcePlan

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
    prefetch: bool = False
    memoryObserver: Callable[[BQMExecutionMemory], None] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.batchSize, bool)
            or not isinstance(self.batchSize, Integral)
            or self.batchSize <= 0
        ):
            raise ValueError("batchSize must be a positive integer")
        if self.maxBatchBytes is not None and self.maxBatchBytes <= 0:
            raise ValueError("maxBatchBytes must be positive or None")
        if not isfinite(self.memoryMultiplier) or self.memoryMultiplier < 1.0:
            raise ValueError("memoryMultiplier must be at least one")
        if not isinstance(self.prefetch, bool):
            raise TypeError("prefetch must be a boolean")

    def execute(
        self,
        solver: BQMSolver,
        items: Iterable[tuple[Context, QUBOProblem]],
        solverParameters: Mapping[str, Any] | None = None,
    ) -> Iterator[tuple[Context, BQMOptimizationResult]]:
        memory = ExecutionMemoryTracker(self.memoryObserver)
        batches = self._batches(solver, items, solverParameters, memory)
        solver.beginSeries()
        try:
            if self.prefetch:
                # A single producer owns the iterator. At most one future batch
                # is retained while the caller solves and decodes the current one.
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="qubo-prefetch") as executor:
                    future = executor.submit(next, batches, None)
                    while (batch := future.result()) is not None:
                        future = executor.submit(next, batches, None)
                        yield from self._solve(solver, batch, solverParameters, memory)
            else:
                for batch in batches:
                    yield from self._solve(solver, batch, solverParameters, memory)
        finally:
            try:
                batches.close()
            finally:
                memory.update(finished=True)
                solver.endSeries()

    @staticmethod
    def _solve(
        solver: BQMSolver,
        batch: tuple[list[tuple[Context, QUBOProblem]], BQMResourcePlan],
        solverParameters: Mapping[str, Any] | None,
        memory: ExecutionMemoryTracker,
    ) -> Iterator[tuple[Context, BQMOptimizationResult]]:
        items, plan = batch
        memory.update(active=plan.hostBytes)
        try:
            contexts, problems = zip(*items)
            results = solver.solvePlanned(problems, plan, solverParameters)
            if len(results) != len(items):
                raise RuntimeError("BQM solver returned an incomplete batch")
            yield from zip(contexts, results)
        finally:
            memory.update(retainedDelta=-plan.hostBytes, active=0)

    def _batches(
        self,
        solver: BQMSolver,
        items: Iterable[tuple[Context, QUBOProblem]],
        solverParameters: Mapping[str, Any] | None,
        memory: ExecutionMemoryTracker,
    ) -> Iterator[tuple[list[tuple[Context, QUBOProblem]], BQMResourcePlan]]:
        iterator = iter(items)
        pending: tuple[Context, QUBOProblem] | None = None
        parallelism = min(solver.batchParallelism, self.batchSize)
        if parallelism <= 0:
            raise ValueError("BQM solver batchParallelism must be positive")
        try:
            while True:
                batch: list[tuple[Context, QUBOProblem]] = []
                sizes: list[int] = []
                while len(batch) < self.batchSize:
                    try:
                        if pending is not None:
                            item = pending
                        else:
                            item = next(iterator)
                            memory.update(retainedDelta=item[1].numericMemoryBytes)
                    except StopIteration:
                        break
                    pending = None
                    item_bytes = max(
                        int(item[1].numericMemoryBytes * self.memoryMultiplier),
                        solver.estimatedWorkingMemoryBytes(item[1], solverParameters),
                    )
                    if (
                        len(batch) >= parallelism
                        and self._exceedsWorkerBudget(sizes + [item_bytes], parallelism)
                    ):
                        pending = item
                        break
                    batch.append(item)
                    sizes.append(item_bytes)
                if not batch:
                    return
                yield batch, BQMResourcePlan.create(
                    sizes, parallelism, sum(item[1].numericMemoryBytes for item in batch))
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def _exceedsWorkerBudget(self, sizes: list[int], parallelism: int) -> bool:
        return not BQMResourcePlan.create(sizes, parallelism).fits(self.maxBatchBytes)
