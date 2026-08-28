from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import dimod
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backtest_orchestrator import (  # noqa: E402
    BacktestConfig,
    repair_one_hot,
    run_backtest,
)
from market_to_qubo import build_qubos  # noqa: E402
from qubo_model import CompactQubo  # noqa: E402
from sbm_torch import SBMConfig, TorchBatchSBMSolver  # noqa: E402


class TorchBatchSolverTests(unittest.TestCase):
    def test_batching_does_not_change_seeded_results(self) -> None:
        first = dimod.BinaryQuadraticModel(
            {0: -0.7, 1: 0.2, 2: -0.1},
            {(0, 1): 0.5, (1, 2): -0.4},
            0.0,
            dimod.BINARY,
        )
        second = dimod.BinaryQuadraticModel(
            {0: 0.4, 1: -0.8}, {(0, 1): 0.3}, 0.2, dimod.BINARY
        )
        config = SBMConfig(steps=25, runs=3, dt=0.1, dtype="float64")
        solver = TorchBatchSBMSolver("cpu")
        batched = solver.solve_batch([first, second], config, [11, 29])
        separate = [
            solver.solve_batch([first], config, [11])[0],
            solver.solve_batch([second], config, [29])[0],
        ]
        for actual, expected in zip(batched, separate):
            np.testing.assert_array_equal(actual.sample, expected.sample)
            self.assertAlmostEqual(actual.energy, expected.energy, places=12)
            self.assertEqual(actual.raw_run, expected.raw_run)


class QuboTests(unittest.TestCase):
    def test_portfolio_overlay_changes_only_linear_coefficients(self) -> None:
        asset_grids = [
            {
                "z": np.array([-1.0, 1.0]),
                "log_return": np.array([-0.1, 0.1]),
                "simple_return": np.expm1(np.array([-0.1, 0.1])),
            },
            {
                "z": np.array([-0.5, 0.5]),
                "log_return": np.array([-0.05, 0.05]),
                "simple_return": np.expm1(np.array([-0.05, 0.05])),
            },
        ]
        neighbors = np.array([[1], [0]], dtype=np.int64)
        correlations = np.array([[0.3], [0.3]])
        arguments = (
            asset_grids,
            np.array([0.6, -0.4]),
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            neighbors,
            correlations,
            2.0,
            0.1,
        )
        total, _, _ = build_qubos(*arguments, numeric_labels=True)
        structural, _, _ = build_qubos(
            asset_grids,
            np.zeros(2),
            *arguments[2:],
            numeric_labels=True,
        )
        total_vectors = total.to_numpy_vectors(variable_order=range(4))
        structural_vectors = structural.to_numpy_vectors(variable_order=range(4))
        np.testing.assert_array_equal(
            total_vectors.quadratic.row_indices,
            structural_vectors.quadratic.row_indices,
        )
        np.testing.assert_array_equal(
            total_vectors.quadratic.col_indices,
            structural_vectors.quadratic.col_indices,
        )
        np.testing.assert_allclose(
            total_vectors.quadratic.biases,
            structural_vectors.quadratic.biases,
        )
        expected_overlay = np.concatenate(
            [0.6 * asset_grids[0]["simple_return"], -0.4 * asset_grids[1]["simple_return"]]
        )
        np.testing.assert_allclose(
            total_vectors.linear_biases - structural_vectors.linear_biases,
            expected_overlay,
        )

        compact, _, _ = build_qubos(*arguments, compact=True)
        self.assertIsInstance(compact, CompactQubo)
        samples = np.array(
            [[(value >> bit) & 1 for bit in range(4)] for value in range(16)],
            dtype=np.uint8,
        )
        np.testing.assert_allclose(
            compact.energies(samples),
            total.energies((samples, range(4))),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            compact.to_dimod().energies((samples, range(4))),
            total.energies((samples, range(4))),
            rtol=0.0,
            atol=1e-12,
        )

    def test_repair_enforces_exactly_one_state_per_group(self) -> None:
        bqm = dimod.BinaryQuadraticModel(
            {0: -1.0, 1: -0.5, 2: 0.1, 3: -0.2},
            {(0, 1): 2.0, (2, 3): 2.0, (0, 2): 0.3},
            0.0,
            dimod.BINARY,
        )
        repaired, violations, _ = repair_one_hot(
            bqm,
            np.array([1, 1, 0, 0], dtype=np.uint8),
            np.array([0, 2, 4], dtype=np.int64),
            sweeps=1,
        )
        self.assertEqual(violations, 2)
        self.assertEqual(int(repaired[:2].sum()), 1)
        self.assertEqual(int(repaired[2:].sum()), 1)


class EndToEndTests(unittest.TestCase):
    def test_one_day_uses_only_prior_returns_and_writes_signed_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "assets_00004"
            folder.mkdir()
            dates = pd.bdate_range("2024-01-01", periods=35)
            rng = np.random.default_rng(123)
            returns = rng.normal(0.0002, 0.01, size=(len(dates) - 1, 4))
            prices = np.vstack((np.full(4, 100.0), 100.0 * np.exp(np.cumsum(returns, axis=0))))
            frame = pd.DataFrame(prices, index=dates, columns=[str(i) for i in range(4)])
            frame.index.name = "date"
            frame.iloc[:30].to_csv(folder / "historical_close.csv")
            frame.iloc[30:].to_csv(folder / "backtest_close.csv")
            pd.DataFrame(
                {
                    "client_id": ["p"] * 4,
                    "ticker": [str(i) for i in range(4)],
                    "weight": [0.3, 0.2, -0.25, -0.25],
                }
            ).to_csv(folder / "portfolio.csv", index=False)
            output = Path(temporary) / "results.csv"
            config = BacktestConfig(
                subfolder=folder,
                output=output,
                window=20,
                grid_points=(3, 3),
                scenario_indices=(0, 4),
                z_bins=3,
                nearest=10,
                top_k_neighbors=2,
                device="cpu",
                correlation_device="cpu",
                scenario_batch_size=2,
                day_limit=1,
                decode_sweeps=1,
                sbm=SBMConfig(steps=8, runs=2, dt=0.1, dtype="float64"),
            )
            result = run_backtest(config)
            self.assertEqual(len(result), 1)
            row = result.iloc[0]
            self.assertEqual(int(row["calibration_returns"]), 20)
            self.assertLess(pd.Timestamp(row["calibration_end"]), pd.Timestamp(row["date"]))
            self.assertEqual(int(row["scenarios"]), 2)
            expected_error = (
                float(row["margin"]) + float(row["realized_pnl"])
            ) / float(row["gross_exposure"])
            self.assertAlmostEqual(float(row["signed_margin_error"]), expected_error)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".summary.json").is_file())

            # Alter the return being revealed and every later close.  The
            # issued margin must be unchanged because those prices were not in
            # the information set available before the evaluation date.
            changed = pd.read_csv(folder / "backtest_close.csv", index_col="date")
            changed.iloc[1:] *= 1.25
            changed.to_csv(folder / "backtest_close.csv")
            changed_result = run_backtest(
                replace(config, output=Path(temporary) / "changed.csv")
            )
            self.assertAlmostEqual(
                float(changed_result.iloc[0]["margin"]), float(row["margin"]), places=12
            )
            self.assertNotAlmostEqual(
                float(changed_result.iloc[0]["realized_pnl"]),
                float(row["realized_pnl"]),
                places=8,
            )


if __name__ == "__main__":
    unittest.main()
