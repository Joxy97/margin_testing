"""Tests for BQM solver interfaces and factories."""

import unittest
from collections.abc import Mapping
from typing import Any

import dimod
import numpy

from margin_calculator.optimization.optimization_solver.bqm_solver import (
    BQMSolver,
    BQMSolverFactory,
    BQMOptimizationResult,
    OptimizationProblem,
    OptimizationSolver,
    OptimizationSolverResult,
    PlanarGraphBQMSolver,
    QUBOProblem,
    RandomBQMSolver,
    SBMBQMSolver,
    SimulatedAnnealingBQMSolver,
    SteepestDescentBQMSolver,
    TabuBQMSolver,
    TreeDecompositionBQMSolver,
    TreeDecompositionSamplerBQMSolver,
)


class StubBQMSolver(BQMSolver):
    def __init__(self, answer: str = "solved") -> None:
        self.answer = answer
        self.problem: QUBOProblem | None = None
        self.solverParameters: dict[str, Any] = {}

    def solve(
        self,
        problem: QUBOProblem,
        solverParameters: Mapping[str, Any] | None = None,
    ) -> BQMOptimizationResult:
        self.problem = problem
        self.solverParameters = dict(solverParameters or {})
        return BQMOptimizationResult({"x": 1}, -1.0)


class BQMSolverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        BQMSolverFactory.registerSolver("stub", StubBQMSolver)

    def test_solver_receives_a_qubo_problem(self) -> None:
        problem = QUBOProblem(
            linear=numpy.array([-1.0]),
            quadraticHeads=numpy.array([], dtype=numpy.uint32),
            quadraticTails=numpy.array([], dtype=numpy.uint32),
            quadraticBiases=numpy.array([]),
        )
        solver = StubBQMSolver()

        result = solver.solve(problem, {"num_reads": 10})

        self.assertIsInstance(result, OptimizationSolverResult)
        self.assertIsInstance(result, BQMOptimizationResult)
        self.assertIs(solver.problem, problem)
        self.assertEqual(solver.solverParameters, {"num_reads": 10})
        self.assertIsInstance(problem, OptimizationProblem)
        self.assertIsInstance(solver, OptimizationSolver)

    def test_factory_creates_registered_solver_with_parameters(self) -> None:
        solver = BQMSolverFactory.createBQMSolver(
            "stub",
            {"answer": "custom"},
        )

        self.assertIsInstance(solver, StubBQMSolver)
        self.assertEqual(solver.answer, "custom")

    def test_factory_rejects_unknown_solver(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown BQM solver"):
            BQMSolverFactory.createBQMSolver("unknown")

    def test_factory_includes_all_bqm_solvers(self) -> None:
        expected_solvers = {
            "planar_graph": PlanarGraphBQMSolver,
            "random": RandomBQMSolver,
            "sbm": SBMBQMSolver,
            "simulated_annealing": SimulatedAnnealingBQMSolver,
            "steepest_descent": SteepestDescentBQMSolver,
            "tabu": TabuBQMSolver,
            "tree_decomposition_solver": TreeDecompositionBQMSolver,
            "tree_decomposition_sampler": TreeDecompositionSamplerBQMSolver,
        }

        for name, solver_class in expected_solvers.items():
            self.assertIsInstance(
                BQMSolverFactory.createBQMSolver(name),
                solver_class,
            )

    def test_solver_selects_first_valid_one_hot_sample_by_energy(self) -> None:
        sample_set = dimod.SampleSet.from_samples(
            [
                {
                    0: 1,
                    1: 1,
                    2: 0,
                    3: 0,
                },
                {
                    0: 1,
                    1: 0,
                    2: 0,
                    3: 1,
                },
            ],
            vartype=dimod.BINARY,
            energy=[-10.0, -5.0],
        )

        problem = QUBOProblem(
            numpy.zeros(4),
            numpy.array([], dtype=numpy.uint32),
            numpy.array([], dtype=numpy.uint32),
            numpy.array([]),
            oneHotGroups=((0, 1), (2, 3)),
        )
        sample, energy = StubBQMSolver._selectBestSample(sample_set, problem)

        self.assertEqual(energy, -5.0)
        self.assertEqual(sample[0] + sample[1], 1)
        self.assertEqual(sample[2] + sample[3], 1)

    @unittest.skipUnless(
        SBMBQMSolver().libraryPath.is_file(),
        "build the sbm_python CMake target to run this integration test",
    )
    def test_sbm_solver_uses_the_cpp_kernel(self) -> None:
        problem = QUBOProblem(
            numpy.array([-1.0, 0.5]),
            numpy.array([0], dtype=numpy.uint32),
            numpy.array([1], dtype=numpy.uint32),
            numpy.array([-0.25]),
        )
        solver = BQMSolverFactory.createBQMSolver("sbm")

        result = solver.solve(
            problem,
            {"steps": 100, "runs": 4, "seed": 7},
        )

        self.assertAlmostEqual(result.energy, problem.energy(result.sample))
        self.assertEqual(len(result.sample), 2)

    @unittest.skipUnless(
        SBMBQMSolver().libraryPath.is_file(),
        "build the sbm_python CMake target to run this integration test",
    )
    def test_sbm_solver_batches_different_problem_sizes(self) -> None:
        problems = [
            QUBOProblem(
                numpy.array([-1.0, 0.5]),
                numpy.array([0], dtype=numpy.uint32),
                numpy.array([1], dtype=numpy.uint32),
                numpy.array([-0.25]),
            ),
            QUBOProblem(
                numpy.array([0.2, -0.4, -0.1]),
                numpy.array([0, 1], dtype=numpy.uint32),
                numpy.array([1, 2], dtype=numpy.uint32),
                numpy.array([0.3, -0.2]),
            ),
        ]

        results = SBMBQMSolver().solveMany(
            problems,
            {"steps": 100, "runs": 3, "seed": 9},
        )

        self.assertEqual(len(results), 2)
        for problem, result in zip(problems, results):
            self.assertEqual(len(result.sample), problem.variableCount)
            self.assertAlmostEqual(result.energy, problem.energy(result.sample))

    def test_sbm_parameters_are_validated_before_native_allocation(self) -> None:
        with self.assertRaisesRegex(ValueError, "steps and runs"):
            SBMBQMSolver._getParameters({"runs": 0})

    def test_numeric_qubo_arrays_are_immutable(self) -> None:
        problem = QUBOProblem(
            numpy.array([-1.0]),
            numpy.array([], dtype=numpy.uint32),
            numpy.array([], dtype=numpy.uint32),
            numpy.array([]),
        )

        with self.assertRaises(ValueError):
            problem.linear[0] = 2.0

    def test_base_solver_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            BQMSolver()


if __name__ == "__main__":
    unittest.main()
