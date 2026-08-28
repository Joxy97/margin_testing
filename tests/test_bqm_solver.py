"""Tests for BQM solver interfaces and factories."""

import unittest

import dimod

from decision_maker.optimization.optimization_solver.bqm_solver import (
    BQMSolver,
    BQMSolverFactory,
    OptimizationProblem,
    OptimizationSolver,
    OptimizationSolverResult,
    QUBOProblem,
)


class StubBQMSolver(BQMSolver):
    def __init__(self, answer: str = "solved") -> None:
        self.answer = answer
        self.problem: QUBOProblem | None = None

    def solve(self, problem: QUBOProblem) -> OptimizationSolverResult:
        self.problem = problem
        return OptimizationSolverResult()


class BQMSolverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        BQMSolverFactory.registerSolver("stub", StubBQMSolver)

    def test_solver_receives_a_qubo_problem(self) -> None:
        problem = QUBOProblem.from_ising({"x": -1}, {})
        solver = StubBQMSolver()

        result = solver.solve(problem)

        self.assertIsInstance(result, OptimizationSolverResult)
        self.assertIs(solver.problem, problem)
        self.assertIsInstance(problem, dimod.BinaryQuadraticModel)
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

    def test_base_solver_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            BQMSolver()


if __name__ == "__main__":
    unittest.main()
