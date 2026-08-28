"""Tests for BQM solver interfaces and factories."""

import unittest
from typing import Any

import dimod

from bqmsolver import BQMSolver, BQMSolverFactory


class StubBQMSolver(BQMSolver):
    def __init__(self, answer: str = "solved") -> None:
        self.answer = answer

    def solve(self, bqm: dimod.BinaryQuadraticModel) -> Any:
        return self.answer, bqm


class BQMSolverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        BQMSolverFactory.registerSolver("stub", StubBQMSolver)

    def test_solver_receives_a_binary_quadratic_model(self) -> None:
        bqm = dimod.BinaryQuadraticModel.from_ising({"x": -1}, {})
        solver = StubBQMSolver()

        answer, solved_bqm = solver.solve(bqm)

        self.assertEqual(answer, "solved")
        self.assertIs(solved_bqm, bqm)

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
