"""Offline caller contracts for library parameter banks and local Ocean files."""

from itertools import product
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import dimod
import numpy as np
import torch

from qubo_solvers import create_bqm_solver

with patch.object(sys, "path", [str(Path(__file__).parents[1] / "tools"), *sys.path]):
    import benchmark_biqmac as biqmac
    import benchmark_biqmac_bandit as bandit
    import benchmark_ocean_bqms as ocean
    from benchmark_qoblib_sac_comparison import supplied_parameters
    from qoblib_hybrid_sac import RANGES, branches, decode
    from prepare_qoblib_qubos import memory_estimates


class SolverToolMigrationTest(unittest.TestCase):
    def test_preparation_counts_dense_canonical_solver_workspace(self):
        # Sparse input no longer makes the canonical SVL/TRF workspace sparse.
        small, large = memory_estimates(100, 0), memory_estimates(1000, 0)
        self.assertGreater(large["SVL"], 56*1000*1000)
        self.assertGreater(large["TRF"], 56*1000*1000)
        self.assertGreater(large["TRF"]/small["TRF"], 20)
        self.assertLess(large["SBM"], large["TRF"])

    def test_parameter_banks_match_canonical_solver_contracts(self):
        banks = bandit.actionBank(17, 32)
        for label, solver_id in biqmac.SOLVERS.items():
            solver = create_bqm_solver(solver_id, {"device": "cpu"})
            for arm in banks[label]:
                with self.subTest(solver=solver_id, arm=arm):
                    solver._getParameters(biqmac.parameters(label, 2, 3, 17) | arm)
            for branch in range(len(branches(label))):
                action = (branch, np.zeros(len(RANGES[label])))
                with self.subTest(solver=solver_id, branch=branch):
                    solver._getParameters(supplied_parameters(label, decode(label, action), 17))

    def test_ocean_local_file_preserves_objective_and_invokes_three_solvers(self):
        model = dimod.BinaryQuadraticModel(
            {"x_1_1": -2., "x_0_1": 1., "x_1_0": .5, "x_0_0": -1.},
            {("x_0_0", "x_1_1"): 1.25, ("x_0_1", "x_1_0"): -.75},
            3.125, dimod.BINARY,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario_1.bqm"
            with model.to_file() as stream:
                path.write_bytes(stream.read())
            problem, grouped, elapsed = ocean.load_problem(path)
            self.assertGreaterEqual(elapsed, 0)
            self.assertEqual(tuple(tuple(g) for g in grouped.iterOneHotGroups()), ((0, 1), (2, 3)))
            for sample in product((0, 1), repeat=4):
                labels = dict(zip(("x_0_0", "x_0_1", "x_1_0", "x_1_1"), sample))
                self.assertEqual(problem.energy(sample), model.energy(labels))
                self.assertEqual(grouped.energy(sample), model.energy(labels))
            records = ocean.benchmark(path, "cpu", 2, 2, 2, 17, 1.)
            self.assertEqual({r["solver"] for r in records},
                             {"lib_greedy_local_search", "lib_simulated_annealing", "lib_simulated_bifurcation"})
            for row in records:
                self.assertEqual(row["one_hot_violations"], 0)
                self.assertTrue(np.isfinite(row["energy"]))
                self.assertGreaterEqual(row["repair_seconds"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_ocean_gpu_scoring_uses_the_same_local_file_format(self):
        model = dimod.BinaryQuadraticModel({"x_0_0": -1., "x_0_1": 2.}, {}, 4., dimod.BINARY)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario_1.bqm"
            with model.to_file() as stream:
                path.write_bytes(stream.read())
            for row in ocean.benchmark(path, "cuda:0", 2, 2, 2, 17, 1.):
                self.assertEqual(row["one_hot_violations"], 0)
                self.assertIn(row["energy"], (3., 6.))


if __name__ == "__main__":
    unittest.main()
