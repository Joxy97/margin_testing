from __future__ import annotations

import json
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
    ScenarioData,
    ScenarioProblem,
    decode_hard_plausible,
    greedy_baseline_pnl,
    repair_one_hot,
    rolling_ew_pca,
    run_backtest,
)
from market_to_qubo import (  # noqa: E402
    PlausibilityModel,
    build_qubos,
    fit_ew_pca,
    fit_plausibility_model,
    plausibility_score,
)
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

    def test_all_raw_candidates_are_repaired_before_selection(self) -> None:
        model = CompactQubo(
            linear=np.array([-2.0, -1.0, -2.0, -1.0]),
            heads=np.array([0, 2], dtype=np.int32),
            tails=np.array([1, 3], dtype=np.int32),
            quadratic=np.array([0.1, 0.1]),
        )
        config = SBMConfig(steps=10, runs=5, dt=0.1, dtype="float64")
        result = TorchBatchSBMSolver("cpu").solve_batch(
            [model],
            config,
            [17],
            [np.array([0, 2, 4], dtype=np.int64)],
        )[0]
        self.assertEqual(result.candidate_count, 5)
        self.assertGreaterEqual(result.raw_feasible_candidates, 0)
        self.assertLessEqual(result.raw_feasible_candidates, 5)
        self.assertEqual(int(result.sample[:2].sum()), 1)
        self.assertEqual(int(result.sample[2:].sum()), 1)
        self.assertAlmostEqual(result.energy, model.energy(result.sample))
        self.assertAlmostEqual(result.raw_energy, model.energy(result.raw_sample))


class QuboTests(unittest.TestCase):
    def test_greedy_baseline_selects_each_assets_worst_state(self) -> None:
        data = ScenarioData(
            scenario_index=7,
            shocks=None,
            asset_grids=[],
            group_offsets=np.array([0, 3, 6], dtype=np.int64),
            portfolio_linear=np.array(
                [0.4, -0.2, 0.1, -0.3, -0.5, 0.8], dtype=np.float64
            ),
        )
        self.assertAlmostEqual(greedy_baseline_pnl(data), -0.7)

    def test_neighbor_graph_uses_union_and_deduplicates_mutual_pairs(self) -> None:
        asset_grids = [
            {
                "z": np.array([-1.0, 1.0]),
                "log_return": np.array([-0.1, 0.1]),
                "simple_return": np.expm1(np.array([-0.1, 0.1])),
            }
            for _ in range(3)
        ]
        # The 2 -> 0 nomination used to be dropped solely because 0 < 2.
        neighbors = np.array([[2], [2], [0]], dtype=np.int64)
        correlations = np.array([[0.4], [-0.3], [0.4]])
        compact, _, compatibility_edges = build_qubos(
            asset_grids,
            np.ones(3),
            np.zeros(3),
            np.ones(3),
            neighbors,
            correlations,
            1.0,
            0.1,
            compact=True,
        )
        self.assertIsInstance(compact, CompactQubo)
        self.assertEqual(compatibility_edges, 2)
        # Three one-hot edges plus two 2x2 compatibility blocks.
        self.assertEqual(compact.num_interactions, 3 + 2 * 4)

    def test_qubo_is_homogeneous_in_portfolio_exposure(self) -> None:
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
        scaled_arguments = (asset_grids, 37.0 * arguments[1], *arguments[2:])
        scaled, _, _ = build_qubos(*scaled_arguments, numeric_labels=True)
        total_vectors = total.to_numpy_vectors(variable_order=range(4))
        scaled_vectors = scaled.to_numpy_vectors(variable_order=range(4))
        np.testing.assert_array_equal(
            total_vectors.quadratic.row_indices,
            scaled_vectors.quadratic.row_indices,
        )
        np.testing.assert_array_equal(
            total_vectors.quadratic.col_indices,
            scaled_vectors.quadratic.col_indices,
        )
        np.testing.assert_allclose(
            total_vectors.quadratic.biases,
            scaled_vectors.quadratic.biases,
        )
        np.testing.assert_allclose(
            total_vectors.linear_biases,
            scaled_vectors.linear_biases,
        )
        self.assertAlmostEqual(total_vectors.offset, scaled_vectors.offset)

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

    def test_safe_penalty_excludes_infeasible_global_minima(self) -> None:
        asset_grids = [
            {
                "z": np.array([-1.0, 1.0]),
                "log_return": np.array([-0.5, 0.5]),
                "simple_return": np.expm1(np.array([-0.5, 0.5])),
            }
            for _ in range(2)
        ]
        model, _, _ = build_qubos(
            asset_grids,
            np.array([0.5, -0.5]),
            np.zeros(2),
            np.ones(2),
            np.array([[1], [0]], dtype=np.int64),
            np.array([[0.8], [0.8]]),
            0.0,
            1.0,
            compact=True,
        )
        samples = np.array(
            [[(value >> bit) & 1 for bit in range(4)] for value in range(16)],
            dtype=np.uint8,
        )
        energies = model.energies(samples)
        minimizers = samples[np.isclose(energies, energies.min(), atol=1e-12)]
        self.assertTrue(np.all(minimizers[:, :2].sum(axis=1) == 1))
        self.assertTrue(np.all(minimizers[:, 2:].sum(axis=1) == 1))

    def test_psd_plausibility_is_permutation_invariant(self) -> None:
        rng = np.random.default_rng(72)
        samples = rng.normal(size=(80, 5))
        samples[:, 3] += 0.65 * samples[:, 1]
        residual = rng.normal(size=5)
        permutation = np.array([3, 0, 4, 1, 2])
        first = fit_plausibility_model(samples, top_k=4, confidence=0.95)
        second = fit_plausibility_model(
            samples[:, permutation], top_k=4, confidence=0.95
        )
        self.assertAlmostEqual(first.score(residual), second.score(residual[permutation]))
        self.assertAlmostEqual(first.threshold, second.threshold)

    def test_hard_decoder_never_accepts_an_implausible_sample(self) -> None:
        plausibility = PlausibilityModel(
            residual_mean=np.zeros(2),
            residual_scale=np.ones(2),
            edge_heads=np.empty(0, dtype=np.int64),
            edge_tails=np.empty(0, dtype=np.int64),
            edge_weights=np.empty(0),
            threshold=1.0,
            confidence=0.95,
        )
        states = (np.array([-1.0, 0.0, 1.0]),) * 2
        problem = ScenarioProblem(
            scenario_index=0,
            bqm=CompactQubo(
                linear=np.zeros(6),
                heads=np.empty(0, dtype=np.int32),
                tails=np.empty(0, dtype=np.int32),
                quadratic=np.empty(0),
            ),
            group_offsets=np.array([0, 3, 6], dtype=np.int64),
            portfolio_linear=np.array([-2.0, 0.0, 1.0, -3.0, 0.0, 1.0]),
            gross_exposure=1.0,
            plausibility=plausibility,
            plausibility_states=states,
        )
        raw = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)
        decoded, pnl, score = decode_hard_plausible(problem, raw)
        self.assertLessEqual(score, plausibility.threshold)
        self.assertAlmostEqual(score, plausibility_score(states, np.array([0, 1]), plausibility))
        self.assertAlmostEqual(pnl, -2.0)
        self.assertEqual(int(decoded[:3].sum()), 1)
        self.assertEqual(int(decoded[3:].sum()), 1)

    def test_exporter_and_backtester_share_identical_ew_pca(self) -> None:
        rng = np.random.default_rng(91)
        frame = pd.DataFrame(
            rng.normal(0.0, 0.01, size=(30, 5)),
            columns=[f"a{i}" for i in range(5)],
        )
        scaler, z_values, _, exported = fit_ew_pca(
            frame.to_numpy(),
            np.empty((0, 5)),
            2,
            0.93,
            device_name="cpu",
            dtype_name="float64",
        )
        prepared, rolling = rolling_ew_pca(frame, 0.93, "cpu", "float64")
        np.testing.assert_allclose(prepared.standardizer.mean_, scaler.mean_)
        np.testing.assert_allclose(prepared.standardizer.scale_, scaler.scale_)
        np.testing.assert_allclose(prepared.historical_z, z_values)
        np.testing.assert_allclose(rolling.loadings, exported.loadings)
        np.testing.assert_allclose(rolling.eigenvalues, exported.eigenvalues)
        np.testing.assert_allclose(rolling.factors, exported.factors)

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
            sweeps=100,
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
                decode_sweeps=100,
                sbm=SBMConfig(steps=8, runs=2, dt=0.1, dtype="float64"),
            )
            result = run_backtest(config)
            self.assertEqual(len(result), 1)
            row = result.iloc[0]
            self.assertEqual(int(row["calibration_returns"]), 20)
            self.assertLess(pd.Timestamp(row["calibration_end"]), pd.Timestamp(row["date"]))
            self.assertEqual(int(row["scenarios"]), 2)
            self.assertTrue(bool(row["hard_plausibility_feasible"]))
            self.assertLessEqual(
                float(row["plausibility_score"]),
                float(row["plausibility_threshold"]) + 1e-12,
            )
            self.assertEqual(int(row["solver_candidates"]), config.sbm.runs)
            expected_error = (
                float(row["margin"]) + float(row["realized_pnl"])
            ) / float(row["gross_exposure"])
            self.assertAlmostEqual(float(row["signed_margin_error"]), expected_error)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".summary.json").is_file())
            diagnostics_path = output.with_suffix(".diagnostics.csv")
            self.assertTrue(diagnostics_path.is_file())
            diagnostics = pd.read_csv(diagnostics_path)
            self.assertEqual(len(diagnostics), 1)
            self.assertEqual(int(diagnostics.iloc[0]["scenario_batches"]), 1)
            self.assertEqual(int(diagnostics.iloc[0]["build_workers"]), 1)
            self.assertGreater(float(diagnostics.iloc[0]["peak_ram_mib"]), 0.0)

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

            both_output = Path(temporary) / "both.csv"
            both = run_backtest(replace(config, output=both_output, method="both"))
            self.assertEqual(set(both["method"]), {"qubo", "baseline"})
            self.assertEqual(len(both), 2)
            qubo = both.loc[both["method"] == "qubo"].iloc[0]
            baseline = both.loc[both["method"] == "baseline"].iloc[0]
            self.assertLessEqual(
                float(baseline["worst_scenario_pnl"]),
                float(qubo["worst_scenario_pnl"]) + 1e-12,
            )
            self.assertGreaterEqual(
                float(baseline["margin"]), float(qubo["margin"]) - 1e-12
            )
            self.assertEqual(float(baseline["qubo_build_seconds"]), 0.0)
            self.assertEqual(float(baseline["solve_seconds"]), 0.0)
            summary = json.loads(
                both_output.with_suffix(".summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(summary["methods"]), {"baseline", "qubo"})

            baseline_output = Path(temporary) / "baseline.csv"
            baseline_only = run_backtest(
                replace(config, output=baseline_output, method="baseline")
            )
            self.assertEqual(list(baseline_only["method"]), ["baseline"])
            baseline_diagnostics = pd.read_csv(
                baseline_output.with_suffix(".diagnostics.csv")
            ).iloc[0]
            self.assertEqual(baseline_diagnostics["correlation_device"], "not-used")
            self.assertEqual(int(baseline_diagnostics["steps"]), 0)
            self.assertEqual(int(baseline_diagnostics["runs"]), 0)


if __name__ == "__main__":
    unittest.main()
