"""Opt-in CUDA integration check for radius sweeps with in-worker repair."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

with patch.object(sys, "path", [str(Path(__file__).parents[1]/"tools"), *sys.path]):
    from sweep_factor_stress import SOLVERS, execute
    from benchmark_factor_stress import writeJson
from risk_state_generator.factor_stress_model import FactorStressModel
from margin_calculator.optimization.factor_stress import FactorStressQUBO, FactorStressQUBOConfig, QuadraticStressObjective


@unittest.skipUnless(os.environ.get("RUN_FACTOR_GPU_TESTS") == "1", "opt-in CUDA worker integration")
class FactorSweepWorkerTest(unittest.TestCase):
    def test_radius_sweep_repairs_each_solver_using_its_original_constraints(self):
        model = FactorStressModel(("A", "B", "C"), [.7, -.3, .2], [.01, -.02, 0.],
                                  [[.05, .02, 0.], [.01, -.04, .02], [.03, .01, .04]])
        objective = QuadraticStressObjective(*model.quadraticCoefficients())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for directory in ("models", "samples", "batches"):
                (output/directory).mkdir()
            np.savez(output/"models/000.npz", instruments=model.instruments, exposures=model.exposures,
                     center=model.center, directions=model.directions, constant=objective.constant,
                     gradient=objective.gradient, hessian=objective.hessian)
            writeJson(output/"days.json", dict(days=[dict(index=0, date="2025-07-08", realized_pnl=-.1,
                                                         realized_loss=.1, lattice={})]))
            work = []
            for index, solver in enumerate(SOLVERS):
                trial = dict(id=solver, solver=solver, day=0, bits=8, multiplier=10.**index,
                             radius=2.+index, repeat=0, seed_offset=100)
                work.append(dict(id=index, solver=solver, trials=[trial]))
            execute(SimpleNamespace(output=output, steps=16, runs=2, seed=7, radius=1., devices=[0],
                                    repair_samples=True), work)
            for index, solver in enumerate(SOLVERS):
                row = json.loads((output/"batches"/f"{index:04d}.json").read_text())["rows"][0]
                sample = np.load(output/"repaired_samples"/(solver+".npy"))
                encoded = FactorStressQUBO.build(objective, FactorStressQUBOConfig(8, 2.+index, 1.1, 10.**index))
                self.assertEqual(row["radius"], 2.+index)
                self.assertTrue(encoded.diagnostics(sample)["encoding_feasible"])
                self.assertAlmostEqual(row["repaired_pnl"], float(model.pnl(encoded.coordinates(sample))))
                self.assertEqual(row["repaired_breach"], .1 > row["repaired_margin"])


if __name__ == "__main__":
    unittest.main()
