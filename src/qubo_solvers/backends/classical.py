"""BQMSolver adapters for the classical ``dwave-samplers`` package."""

from collections.abc import Mapping
from importlib import import_module
from typing import Any, ClassVar

from .result import BQMOptimizationResult

from .problem import QUBOProblem
from .base import BQMSolver


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
