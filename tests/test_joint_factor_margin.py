"""Application integration parity, temporal contracts and artifact lifecycle."""

from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from joint_factor_margin import ApplicationConfig, BacktestRunner, CalculationRequest, ExposurePortfolio, InfeasibleSolutionError
from joint_factor_margin.search import SingleAnchorSearch, BoundaryAwareSearch
from joint_factor_margin.reporting import writeJson, writeBacktest
from margin_calculator.optimization import (
    FactorStressQUBO, QuadraticStressObjective, FactorStressRepair, solveOptionFactorStress,
)
from margin_calculator.optimization.optimization_result import BQMOptimizationResult
from margin_calculator.optimization.optimization_solver.bqm_solver import BQMSolver, BatchBQMExecutionPolicy
from option_pricing import VanillaPriceContext
from portfolio import Portfolio
from risk_state_generator import ReturnsPCAGrid, ReturnsPCAKey, FactorStressModel, EuropeanOptionFactorStressModel, OptionFactorStressModel
from risk_state_generator.volatility_forecast import forecastVolatility, rescaleStressModel
from backtesting import BacktestCheckpointStore, MarginBacktester


class AllOnesSolver(BQMSolver):
    def solve(self, problem, solverParameters=None):
        sample = np.ones(problem.variableCount, dtype=np.uint8)
        return BQMOptimizationResult(sample, problem.energy(sample))


class ApplicationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        rng = np.random.default_rng(19)
        self.prices = pd.DataFrame(100*np.exp(np.vstack((np.zeros(3),
            np.cumsum(rng.normal(0., .015, (40, 3)), axis=0)))),
            index=pd.bdate_range("2025-01-01", periods=41), columns=("AAA", "BBB", "CCC"))
        self.prices.to_csv(self.base/"prices.csv", index_label="date")
        self.marginDate = self.prices.index[-1].date()+timedelta(days=1)
        self.document = {
            "schemaVersion": 1, "marginDate": str(self.marginDate),
            "portfolio": {"weights": {"CCC": "1000", "AAA": "2000", "BBB": "-500"}},
            "engine": {
                "downloadManager": {"providers": {"local": "local_csv"},
                                    "requestParameters": {"location": "prices.csv"}},
                "estimation": {"ew_window": 20, "components": 2},
                "encoding": {"bitsPerCoordinate": 3},
                "solver": {"type": "simulated_annealing", "solverParameters": {"seed": 19, "num_reads": 4, "num_sweeps": 50}},
            },
        }

    def config(self, document=None):
        return ApplicationConfig.fromMapping(document or self.document, self.base)

    def model(self, config):
        app = config.createApplication()
        data = app.marketData.acquire(config.request.portfolio, config.request.marginDate)
        scenario = app.scenarios.build(config.request, data)
        return app, scenario.model, data

    def addOptions(self, kind="european"):
        today = self.prices.index[-1].date()
        spots = self.prices.iloc[-1].to_dict()
        style = "A" if kind == "vanilla" else "E"
        mark = float(VanillaPriceContext(100., 180/365., .04, .03, .27, "P", style).evaluate(spots["AAA"]))
        self.document["options"] = {
            "type": kind, "spotPrices": spots, "valuationDate": str(today), "horizonDate": str(self.marginDate),
            "riskFreeRate": .04, "positions": [{"instrumentType": "equity_option", "symbol": "AAA",
                "expirationDate": str(today+timedelta(days=180)), "strike": "100", "optionType": "P",
                "exerciseStyle": style, "quantity": "-2", "multiplier": "100", "dividendYield": .03}],
            "marketPrices": [mark],
        }
        if kind == "vanilla":
            self.document["engine"]["search"] = {"type": "boundary_aware"}

    def test_stock_pipeline_matches_existing_calculation_and_energy(self):
        config = self.config()
        app, model, data = self.model(config)
        grid = ReturnsPCAGrid.construct(ReturnsPCAKey(config.request.portfolio.instruments, 20,
            self.marginDate, .93, 2), data)
        expected = FactorStressModel.fromPCAGrid(grid, config.request.portfolio.toPortfolio())
        np.testing.assert_array_equal(model.directions, expected.directions)
        encoding = FactorStressQUBO.build(QuadraticStressObjective(*expected.quadraticCoefficients()), config.engine.encoding)
        raw = config.engine.solver.createBQMSolver().solve(encoding.problem, config.engine.solver.solverParameters)
        repaired = FactorStressRepair(expected, encoding, config.engine.search.repair).repair(raw.sample)
        result = app.calculate(config.request)
        np.testing.assert_array_equal(result.coordinates, encoding.coordinates(repaired.sample))
        self.assertEqual(result.pnl, float(expected.pnl(result.coordinates)))
        self.assertEqual(result.diagnostics["search"]["sourceEnergy"], encoding.problem.energy(raw.sample))
        self.assertTrue(result.diagnostics["search"]["selected"]["encoding_feasible"])
        self.assertEqual(result.margin, max(0., -repaired.pnl))

    def test_european_pipeline_preserves_existing_mixed_model(self):
        self.addOptions()
        config = self.config()
        app, model, data = self.model(config)
        grid = ReturnsPCAGrid.construct(ReturnsPCAKey(config.request.portfolio.instruments, 20, self.marginDate, .93, 2), data)
        expected = EuropeanOptionFactorStressModel.fromPCAGrid(grid, config.request.portfolio.toPortfolio(), config.request.options)
        for actual, reference in zip(model.quadraticCoefficients(), expected.quadraticCoefficients()):
            np.testing.assert_array_equal(actual, reference)
        result = app.calculate(config.request)
        self.assertEqual(result.pnl, float(expected.pnl(result.coordinates)))

    def test_boundary_search_matches_original_and_batched_anchor_order(self):
        self.addOptions("vanilla")
        config = self.config()
        _, model, _ = self.model(config)
        solver = AllOnesSolver()
        expected = solveOptionFactorStress(model, solver, config=config.engine.encoding)
        # Independently recorded from the pre-integration HEAD implementation.
        self.assertAlmostEqual(expected.pnl, -661.2707918953022, delta=1e-8)
        np.testing.assert_array_equal(expected.coordinates, [1., -2., -2.])
        direct = BoundaryAwareSearch(solver, config.engine.encoding, config.engine.search.repair).search(model)
        batched = BoundaryAwareSearch(solver, config.engine.encoding, config.engine.search.repair,
            executionPolicy=BatchBQMExecutionPolicy(batchSize=3, maxBatchBytes=100000, prefetch=True)).search(model)
        for actual in (direct, batched):
            np.testing.assert_array_equal(actual.coordinates, expected.coordinates)
            self.assertEqual(actual.pnl, expected.pnl)
            self.assertEqual(actual.diagnostics["rawSolverFeasibility"], expected.solverFeasibility)
            self.assertEqual(actual.diagnostics["candidateCount"], len(expected.candidates))

    def test_supplied_vanilla_inputs_preserve_existing_alignment_and_status(self):
        self.addOptions("vanilla")
        self.document["options"]["suppliedVolatilities"] = [.27]
        self.document["localPnlGradient"] = {"AAA": 100., "BBB": -200., "CCC": 300.}
        config = self.config()
        _, model, data = self.model(config)
        grid = ReturnsPCAGrid.construct(ReturnsPCAKey(config.request.portfolio.instruments, 20, self.marginDate, .93, 2), data)
        expected = OptionFactorStressModel.fromPCAGrid(grid, config.request.portfolio.toPortfolio(),
            config.request.options, localPnlGradient=config.request.localPnlGradient)
        np.testing.assert_array_equal(model.equityModel.directions, expected.equityModel.directions)
        self.assertEqual(model.residualAlignment, "supplied")
        self.assertEqual(model.options.calibrations[0].status, "supplied")

    def test_mixed_european_american_book_uses_one_shared_scenario(self):
        self.addOptions("vanilla")
        options = self.document["options"]
        position = deepcopy(options["positions"][0])
        position.update(symbol="BBB", optionType="C", exerciseStyle="E", quantity="1")
        options["positions"].append(position)
        mark = float(VanillaPriceContext(100., 180/365., .04, .03, .27, "C", "E").evaluate(options["spotPrices"]["BBB"]))
        options["marketPrices"].append(mark)
        config = self.config()
        app, model, _ = self.model(config)
        result = app.calculate(config.request)
        self.assertEqual(result.pnl, float(model.pnl(result.coordinates)))
        self.assertAlmostEqual(result.pnl, sum(result.diagnostics["search"]["pnlByType"].values()), delta=1e-9)

    def test_pca_and_overlay_exclude_evaluation_and_future_prices(self):
        self.document["engine"]["volatility"] = {"name": "ewma", "kind": "ewma"}
        config = self.config()
        app, model, data = self.model(config)
        extra = pd.DataFrame({"date": [pd.Timestamp(self.marginDate), pd.Timestamp(self.marginDate)+pd.Timedelta(days=1)],
                              "AAA": [1e5, 2e5], "BBB": [3e5, 4e5], "CCC": [5e5, 6e5]})
        future = app.scenarios.build(config.request, pd.concat([data, extra], ignore_index=True))
        np.testing.assert_array_equal(model.directions, future.model.directions)
        self.assertLess(future.calibrationEnd, self.marginDate)
        grid = ReturnsPCAGrid.construct(ReturnsPCAKey(config.request.portfolio.instruments, 20, self.marginDate, .93, 2), data)
        base = FactorStressModel.fromPCAGrid(grid, config.request.portfolio.toPortfolio())
        prices = data.set_index("date").loc[:, list(config.request.portfolio.instruments)].iloc[-21:]
        overlay = forecastVolatility(np.diff(np.log(prices), axis=0), prices.index, self.marginDate,
            grid.logReturnMean, grid.logReturnScale, config.engine.volatility)
        np.testing.assert_array_equal(model.directions, rescaleStressModel(base, overlay.ratio).directions)

    def test_references_bounds_and_unsupported_option_bounds(self):
        self.document["engine"]["diagnostics"] = {"references": True, "bounds": True, "boundCoordinates": True}
        config = self.config()
        result = config.createApplication().calculate(config.request)
        self.assertEqual(result.diagnostics["references"]["status"], "computed")
        self.assertEqual(result.diagnostics["bounds"]["status"], "computed")
        self.assertGreaterEqual(result.diagnostics["bounds"]["marginBracket"].upper_margin_bound, result.margin)
        self.addOptions()
        config = self.config()
        result = config.createApplication().calculate(config.request)
        self.assertEqual(result.diagnostics["bounds"]["status"], "unsupported")

    def test_infeasible_raw_solution_is_rejected_without_repair(self):
        config = self.config()
        _, model, _ = self.model(config)
        search = SingleAnchorSearch(AllOnesSolver(), config.engine.encoding, config.engine.search.repair, False)
        with self.assertRaises(InfeasibleSolutionError):
            search.search(model)

    def test_requests_results_and_diagnostics_own_their_data(self):
        original = Portfolio({"AAA": Decimal("1")})
        snapshot = ExposurePortfolio.fromPortfolio(original)
        original.weights["AAA"] = Decimal("2")
        self.assertEqual(snapshot.weights["AAA"], Decimal("1"))
        config = self.config()
        result = config.createApplication().calculate(config.request)
        with self.assertRaises(ValueError):
            result.coordinates[0] = 42
        with self.assertRaises(TypeError):
            result.diagnostics["search"]["raw"]["encoding_feasible"] = False
        target = self.base/"result.json"
        writeJson(result, target)
        self.assertEqual(json.loads(target.read_text())["pnl"], result.pnl)

    def test_strict_config_and_unsupported_combinations(self):
        for path, value in (((), {"unknown": 1}), (("engine",), {"unknown": 1}),
                            (("engine", "estimation"), {"ew_window": True}),
                            (("engine", "encoding"), {"radius": float("nan")}),
                            (("engine", "solver"), {"solverParameters": []})):
            document = deepcopy(self.document)
            target = document
            for item in path:
                target = target[item]
            target.update(value)
            with self.assertRaises((ValueError, TypeError)):
                self.config(document)
        self.addOptions()
        self.document["backtest"] = {"dates": [str(self.marginDate)]}
        with self.assertRaisesRegex(ValueError, "option backtesting"):
            self.config()

    def test_duplicate_yaml_keys_and_lookahead_option_snapshot_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate YAML key"):
            ApplicationConfig.fromYamlText("schemaVersion: 1\nschemaVersion: 1\n", self.base)
        self.addOptions()
        self.document["options"]["valuationDate"] = str(self.marginDate)
        with self.assertRaisesRegex(ValueError, "valuationDate must precede"):
            self.config()

    def test_optional_libraries_are_not_imported_by_the_application(self):
        code = "import joint_factor_margin, sys; assert not {'torch', 'yfinance', 'matplotlib'} & sys.modules.keys()"
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_local_input_contents_invalidate_resume_identity(self):
        config = self.config()
        before = config.identity()
        self.prices.iloc[0, 0] += 1
        self.prices.to_csv(self.base/"prices.csv", index_label="date")
        self.assertNotEqual(before, config.identity())

    def test_native_library_path_is_relative_to_yaml_directory(self):
        self.document["engine"]["solver"] = {"type": "sbm", "constructorParameters": {"libraryPath": "native/libsbm_python.so"}}
        config = self.config()
        self.assertEqual(config.engine.solver.constructorParameters["libraryPath"],
                         str(self.base/"native"/"libsbm_python.so"))

    def test_provider_urls_remain_urls_and_are_not_opened_as_local_inputs(self):
        parameters = self.document["engine"]["downloadManager"]["requestParameters"]
        parameters["location"] = "https://example.invalid/prices.csv"
        parameters["revision"] = "fixture-v1"
        config = self.config()
        self.assertEqual(config.engine.downloadManager.requestParameters["location"], parameters["location"])
        self.assertEqual(config.inputFiles, ())
        self.assertEqual(len(config.identity()), 64)

    def test_backtest_parity_resume_csv_and_plot(self):
        dates = tuple(day.date() for day in self.prices.index[-3:])
        self.document["backtest"] = {"dates": [str(day) for day in dates], "confidenceLevel": .99}
        config = self.config()
        app = config.createApplication()
        store = BacktestCheckpointStore(self.base/"checkpoints", config.identity())
        results = BacktestRunner().run(app, config.request.portfolio, config.backtest, checkpointStore=store)
        for actual in results.results["portfolio"].dailyResults:
            expected = MarginBacktester._calculateRealizedPnL(config.request.portfolio.toPortfolio(), actual.date,
                app.marketData.acquire(config.request.portfolio, actual.date))
            self.assertEqual(actual.realizedPnL, expected)
            self.assertEqual(actual.breach, -expected > actual.margin)
        with patch.object(app, "calculate", side_effect=AssertionError("completed days must not run")):
            resumed = BacktestRunner().run(app, config.request.portfolio, config.backtest, checkpointStore=store, resume=True)
        self.assertEqual(resumed.results["portfolio"].dailyResults, results.results["portfolio"].dailyResults)
        paths = writeBacktest(results, self.base/"reports", plots=True)["portfolio"]
        self.assertEqual(len(pd.read_csv(paths.breaches)), 3)
        self.assertTrue(paths.performanceMetrics.exists())
        self.assertTrue((paths.breaches.parent/"coverage.png").exists())

    def test_cli_calculation_and_backtest_resume(self):
        path = self.base/"run.yaml"
        path.write_text(yaml.safe_dump(self.document))
        output = self.base/"calculation.json"
        invocation = [sys.executable, "-m", "joint_factor_margin"]
        completed = subprocess.run(invocation+["calculate", str(path), "--output", str(output)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertGreaterEqual(json.loads(output.read_text())["margin"], 0)
        self.document["backtest"] = {"dates": [str(self.prices.index[-1].date())]}
        path.write_text(yaml.safe_dump(self.document))
        arguments = invocation+["backtest", str(path), "--output", str(self.base/"replay")]
        for extra in ([], ["--resume"]):
            completed = subprocess.run(arguments+extra, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((self.base/"replay"/"manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
