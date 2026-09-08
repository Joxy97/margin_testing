"""Constrained selection through the per-problem public interface."""

import unittest
import numpy

from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem


class CandidateSelectionTest(unittest.TestCase):
    def test_later_feasible_chunk_supersedes_a_lower_energy_repair(self):
        from margin_calculator.optimization.optimization_solver.bqm_solver.candidate_selection import CandidateSelection
        problem = QUBOProblem(linear=[-2., -1.], quadraticHeads=[],
                              quadraticTails=[], quadraticBiases=[], oneHotGroups=((0, 1),))
        selection = CandidateSelection(problem)
        selection.add([((0, 0), -100)])
        self.assertEqual(selection.result(), ((1, 0), -2.))
        selection.add([((0, 1), 100)])
        self.assertEqual(selection.result(), ((0, 1), -1.))

    def test_feasible_candidates_are_ranked_by_source_energy(self):
        from margin_calculator.optimization.optimization_solver.bqm_solver.candidate_selection import CandidateSelection
        problem = QUBOProblem(linear=numpy.array([-2., -1.]), quadraticHeads=[],
                              quadraticTails=[], quadraticBiases=[], oneHotGroups=((0, 1),))
        selection = CandidateSelection(problem)
        selection.add([((1, 0), 100), ((0, 1), -100)])
        self.assertEqual(selection.result(), ((1, 0), -2.))
