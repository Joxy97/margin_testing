"""Constrained selection through the per-problem public interface."""

import unittest
from itertools import product
import numpy

from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem


class CandidateSelectionTest(unittest.TestCase):
    def test_sparse_repair_matches_dense_descent_for_every_binary_sample(self):
        from margin_calculator.optimization.optimization_solver.bqm_solver.candidate_selection import CandidateSelection

        groups = ((5, 0, 3), (6, 2), (7, 1))  # Noncontiguous groups; variable 4 is free.
        rng = numpy.random.default_rng(27)
        for empty in (False, True):
            heads = rng.integers(0, 8, size=50) if not empty else numpy.array([], dtype=int)
            tails = rng.integers(0, 8, size=len(heads))
            biases = rng.integers(-4, 5, size=len(heads)) * .5
            # Include diagonals, duplicate/reversed terms, cancellation and ties.
            p = QUBOProblem(linear=rng.integers(-2, 3, size=8) * .5,
                            quadraticHeads=numpy.concatenate((heads, [0, 0, 3, 0])),
                            quadraticTails=numpy.concatenate((tails, [0, 3, 0, 3])),
                            quadraticBiases=numpy.concatenate((biases, [.5, 1., 1., -2.])),
                            oneHotGroups=groups)
            adjacency, linear = CandidateSelection._repairModel(p)
            dense = numpy.zeros((8, 8))
            dense_linear = p.linear.copy()
            for head, tail, bias in zip(p.quadraticHeads, p.quadraticTails, p.quadraticBiases):
                if head == tail:
                    dense_linear[head] += bias
                else:
                    dense[head, tail] += bias
                    dense[tail, head] += bias
            for sample in product((0, 1), repeat=8):
                expected = numpy.array(sample)
                for _ in range(7):
                    changed = False
                    fields = dense_linear + dense @ expected
                    for group in groups:
                        selected = [variable for variable in group if expected[variable]]
                        previous = selected[0] if len(selected) == 1 else None
                        for variable in selected:
                            expected[variable] = 0
                            fields -= dense[:, variable]
                        chosen = min(group, key=lambda variable: fields[variable])
                        if previous is not None and fields[chosen] >= fields[previous] - 1e-12:
                            chosen = previous
                        expected[chosen] = 1
                        fields += dense[:, chosen]
                        changed |= previous != chosen
                    if not changed:
                        break
                actual, energy = CandidateSelection._repairCandidate(sample, p, groups, adjacency, linear)
                with self.subTest(empty=empty, sample=sample):
                    self.assertEqual(actual, tuple(expected))
                    self.assertEqual(energy, p.energy(expected))
                    self.assertEqual(actual[4], sample[4])

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
