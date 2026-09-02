"""BQMSolver adapters for the classical ``dwave-samplers`` package."""

from collections.abc import Mapping
from importlib import import_module
from typing import Any, ClassVar

from margin_calculator.optimization.optimization_result import (
    BQMOptimizationResult,
)

from ...optimization_problem.qubo_problem import QUBOProblem
from .bqm_solver import BQMSolver
from .bqm_solver_factory import BQMSolverFactory


class _DWaveSamplerBQMSolver(BQMSolver):
    """Adapt a stateless ``dwave.samplers`` sampler to ``BQMSolver``."""

    samplerClassName: ClassVar[str]

    def solve(
        self,
        problem: QUBOProblem,
        solverParameters: Mapping[str, Any] | None = None,
    ) -> BQMOptimizationResult:
        sampler = self._createSampler()
        try:
            import dimod

            bqm = dimod.BinaryQuadraticModel.from_numpy_vectors(
                problem.linear,
                (
                    problem.quadraticHeads,
                    problem.quadraticTails,
                    problem.quadraticBiases,
                ),
                problem.offset,
                dimod.BINARY,
            )
            sample_set = sampler.sample(
                bqm,
                **dict(solverParameters or {}),
            )
            sample, energy = self._selectBestSample(
                sample_set,
                problem,
            )
            return BQMOptimizationResult(
                sample=sample,
                energy=energy,
            )
        finally:
            sampler.close()

    @classmethod
    def _createSampler(cls) -> Any:
        try:
            samplers = import_module("dwave.samplers")
        except ImportError as error:
            raise ImportError(
                "Classical D-Wave solvers require the dwave-samplers package"
            ) from error
        return getattr(samplers, cls.samplerClassName)()


class PlanarGraphBQMSolver(_DWaveSamplerBQMSolver):
    samplerClassName = "PlanarGraphSolver"


class RandomBQMSolver(_DWaveSamplerBQMSolver):
    samplerClassName = "RandomSampler"


class SimulatedAnnealingBQMSolver(_DWaveSamplerBQMSolver):
    samplerClassName = "SimulatedAnnealingSampler"


class SteepestDescentBQMSolver(_DWaveSamplerBQMSolver):
    samplerClassName = "SteepestDescentSolver"


class TabuBQMSolver(_DWaveSamplerBQMSolver):
    samplerClassName = "TabuSampler"


class TreeDecompositionBQMSolver(_DWaveSamplerBQMSolver):
    samplerClassName = "TreeDecompositionSolver"


class TreeDecompositionSamplerBQMSolver(_DWaveSamplerBQMSolver):
    samplerClassName = "TreeDecompositionSampler"


for solver_name, solver_class in {
    "planar_graph": PlanarGraphBQMSolver,
    "random": RandomBQMSolver,
    "simulated_annealing": SimulatedAnnealingBQMSolver,
    "steepest_descent": SteepestDescentBQMSolver,
    "tabu": TabuBQMSolver,
    "tree_decomposition_solver": TreeDecompositionBQMSolver,
    "tree_decomposition_sampler": TreeDecompositionSamplerBQMSolver,
}.items():
    BQMSolverFactory.registerSolver(solver_name, solver_class)
