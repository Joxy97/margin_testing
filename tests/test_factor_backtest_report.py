"""Daily aggregation must not pool seeds as independent market observations."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

with patch.object(sys, "path", [str(Path(__file__).parents[1]/"tools"), *sys.path]):
    from report_factor_backtest import aggregateDaily, summarizeDaily
    from report_factor_sweep import graphSizeRows


class FactorBacktestReportTest(unittest.TestCase):
    def fixtures(self):
        days, trials = [], []
        for date, loss in (("2026-01-05", .03), ("2026-01-06", .04)):
            days.append(dict(date=date, realized_loss=loss, realized_pnl=-loss,
                calibration_start="2025-06-01", calibration_end="2026-01-02",
                references=[dict(method="exact_repricing_continuous", margin=.05)]))
            for solver in ("a", "b"):
                for repeat in (0, 1):
                    trials.append(dict(date=date, solver=solver, repeat=repeat,
                        id=f"{date}_{solver}_{repeat}", repaired_margin=.03 if solver == "b" else .02,
                        realized_loss=loss, totalSeconds=.1, raw_amortized_solve_seconds=.2))
        return days, trials

    def test_daily_worst_margin_and_strict_breach_ignore_seed_multiplicity(self):
        days, trials = self.fixtures()
        daily = aggregateDaily(days, trials, ["a", "b"], 2, {"2026-01-06"})
        combined = [r for r in daily if r["method"] == "combined"]
        self.assertEqual([r["margin"] for r in combined], [.03, .03])
        self.assertEqual([r["breach"] for r in combined], [False, True])
        self.assertTrue(all(r["winner_solver"] == "b" and r["winner_repeat"] == 0 for r in combined))
        summary = summarizeDaily(daily)
        whole = next(r for r in summary if r["method"] == "combined" and r["subset"] == "all_dates")
        self.assertEqual((whole["dates"], whole["breaches"], whole["breach_rate"]), (2, 1, .5))
        selected = next(r for r in summary if r["method"] == "combined" and r["subset"] == "penalty_selection_dates")
        self.assertEqual((selected["dates"], selected["breaches"]), (1, 1))

    def test_incomplete_duplicate_or_mismatched_trials_are_rejected(self):
        days, trials = self.fixtures()
        for invalid in (trials[:-1], trials+[trials[0]], trials[4:]):
            with self.assertRaises(ValueError):
                aggregateDaily(days, invalid, ["a", "b"], 2, set())
        trials[0]["realized_loss"] = 1.
        with self.assertRaises(ValueError):
            aggregateDaily(days, trials, ["a", "b"], 2, set())

    def test_single_penalty_graph_report_does_not_require_zero_penalty(self):
        row = dict(bits=8, multiplier=1e-11, edges=7381, scenario_bits=24,
                   product_bits=84, slack_bits=14, variables=122)
        result = graphSizeRows([row], [8])[0]
        self.assertEqual(result["positive_min_edges"], 7381)
        self.assertIsNone(result["zero_min_edges"])
        row.update(multiplier=0., edges=276)
        result = graphSizeRows([row], [8])[0]
        self.assertEqual(result["zero_max_edges"], 276)
        self.assertIsNone(result["positive_max_edges"])


if __name__ == "__main__":
    unittest.main()
