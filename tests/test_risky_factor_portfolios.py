"""Confidence semantics, deterministic portfolio design and paired comparisons."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import chi2

with patch.object(sys, "path", [str(Path(__file__).parents[1]/"tools"), *sys.path]):
    from risky_factor_portfolios import confidenceRadius, breachUpperBound, compactModel, constructPortfolios, configurations
    from run_risky_factor_backtests import schedule
    from report_risky_factor_backtests import dailyRows
from risk_state_generator.factor_stress_model import FactorStressModel


class RiskyFactorPortfoliosTest(unittest.TestCase):
    def test_confidence_maps_to_radius_not_penalty(self):
        radius = confidenceRadius(.9995)
        self.assertAlmostEqual(radius, 4.2107002064913335)
        self.assertAlmostEqual(chi2.cdf(radius*radius, 3), .9995)
        options = configurations()
        self.assertTrue(all(r["multiplier"] >= 1 for r in options))
        primary = next(r for r in options if r["primary"])
        self.assertEqual(primary["multiplier"], 1.)
        self.assertEqual(primary["repeats"], 3)
        self.assertAlmostEqual(breachUpperBound(0, 294), 1-.05**(1/294))
        self.assertGreater(breachUpperBound(0, 294), .0005)
        for invalid in (0, 1, float("nan"), True):
            with self.assertRaises(ValueError):
                confidenceRadius(invalid)

    def test_compaction_preserves_signed_pnl_gradient_and_taylor_objective(self):
        rng = np.random.default_rng(83)
        model = FactorStressModel(tuple("ABCDE"), [0., .5, -1., 0., .2], rng.normal(0, .01, 5), rng.normal(0, .05, (5, 3)))
        compact = compactModel(model)
        self.assertEqual(compact.instruments, ("B", "C", "E"))
        points = rng.normal(size=(100, 3))
        np.testing.assert_allclose(model.pnl(points), compact.pnl(points), atol=1e-14)
        np.testing.assert_allclose(model.pnlGradient(points[0]), compact.pnlGradient(points[0]), atol=1e-14)
        for a, b in zip(model.quadraticCoefficients(), compact.quadraticCoefficients()):
            np.testing.assert_allclose(a, b, atol=1e-14)

    def test_ten_distinct_fixed_portfolios_have_expected_exposure(self):
        rng = np.random.default_rng(73)
        logs = rng.normal(0, .02, (125, 140))
        names = tuple(f"T{i:03d}" for i in range(140))
        prices = pd.DataFrame(100*np.exp(np.vstack([np.zeros(140), np.cumsum(logs, axis=0)])), columns=names)
        grid = SimpleNamespace(instruments=names, ew_window=125, residuals=logs,
                               logReturnScale=np.ones(140), loadings=rng.normal(size=(2, 140)))
        rows, weights, eligible = constructPortfolios(prices, grid)
        self.assertEqual(len(rows), 10)
        self.assertTrue(eligible.all())
        self.assertEqual(len({row.tobytes() for row in weights}), 10)
        np.testing.assert_allclose(np.abs(weights).sum(axis=1), [1, 1, 1, 1, 1, 1, 2, 2, 4, 3])
        np.testing.assert_allclose(weights.sum(axis=1), [1, 1, 1, 1, -1, -1, 0, 0, 0, -1], atol=1e-14)
        np.testing.assert_array_equal(weights, constructPortfolios(prices, grid)[1])
        with self.assertRaises(ValueError):
            constructPortfolios(pd.concat([prices, prices.iloc[-1:]]), grid)

    def test_schedule_pairs_seed_across_penalty_and_radius(self):
        work = schedule(dict(dates=["a", "b"], settings=configurations()))
        trials = [t for b in work for t in b["trials"]]
        self.assertEqual(len(trials), 48)
        self.assertEqual(len({t["id"] for t in trials}), 48)
        self.assertEqual({t["seed_offset"] for t in trials if t["day"] == 0 and t["repeat"] == 0}, {8000})

    def test_primary_extra_seeds_do_not_leak_into_paired_probe(self):
        config = next(r for r in configurations() if r["primary"])
        day = dict(index=0, date="2025-07-08", realized_loss=.3, realized_pnl=-.3,
                   references=[dict(method="exact_repricing_continuous", coverage=.9995, margin=.6, success=True)])
        trials = [dict(day=0, setting=config["id"], solver=solver, repeat=repeat, id=f"{solver}{repeat}",
                       repaired_margin=.2+.1*repeat)
                  for solver in ("torch_sbm", "torch_svl", "torch_transverse_route") for repeat in range(3)]
        rows = dailyRows([day], trials, [config])
        paired = next(r for r in rows if r["analysis"] == "paired_seed0" and r["solver"] == "combined")
        primary = next(r for r in rows if r["analysis"] == "primary_best3" and r["solver"] == "combined")
        self.assertAlmostEqual(paired["margin"], .2)
        self.assertTrue(paired["breach"])
        self.assertAlmostEqual(primary["margin"], .4)
        self.assertFalse(primary["breach"])


if __name__ == "__main__":
    unittest.main()
