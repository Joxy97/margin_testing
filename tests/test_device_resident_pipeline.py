"""Resident numerical execution through the YAML and MarginEngine seams."""

from importlib.util import find_spec
import tempfile
import unittest
from pathlib import Path

import yaml

from margin_engine import MarginApplicationConfig


@unittest.skipUnless(find_spec("torch"), "requires Torch")
class DeviceResidentPipelineTest(unittest.TestCase):
    def configuration(self, directory, calculator=None):
        root = Path(directory)
        (root / "prices.csv").write_text(
            "date,A\n2024-01-01,100\n2024-01-02,101\n2024-01-03,98\n"
            "2024-01-04,103\n2024-01-05,101\n2024-01-06,105\n"
            "2024-01-07,104\n2024-01-08,108\n2024-01-09,107\n"
            "2024-01-10,110\n2024-01-11,111\n")
        return {
            "marginDate": "2024-01-11", "portfolio": {"weights": {"A": 10}},
            "engine": {
                "numericalExecution": {"type": "torch", "device": "cpu"},
                "downloadManager": {"providers": {"local": "local_csv"},
                                    "requestParameters": {"location": "prices.csv"}},
                "riskStateGenerator": {"ew_window": 5, "components": 1,
                    "scenariosPerComponents": [3], "nZBins": 2, "allowEmptyBinFallback": True},
                "marginCalculator": calculator or {"type": "state_aware_greedy"},
            },
        }

    def test_resident_greedy_matches_the_known_host_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.configuration(directory)
            report = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
        self.assertAlmostEqual(report.margin, 0.1798027685847499, places=12)

    def test_resident_bqm_preserves_paired_margin_and_source_scoring(self):
        calculator = {"type": "bqm", "comparison": {"type": "state_aware_greedy"},
            "solver": {"type": "torch_sbm", "constructorParameters": {"device": "cpu"},
                       "solverParameters": {"steps": 8, "runs": 4, "seed": 13, "dtype": "float64"}},
            "executionPolicy": {"type": "batch", "batchSize": 2}}
        with tempfile.TemporaryDirectory() as directory:
            config = self.configuration(directory, calculator)
            report = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
        self.assertAlmostEqual(report.margin, 0.1798027685847499, places=12)
        self.assertAlmostEqual(report.comparisonMargins["greedy"], 0.1798027685847499, places=12)

    def test_resident_correlation_penalties_match_the_host_reference(self):
        calculator = {"type": "bqm", "modelParameters": {"lambdaCompat": 1.},
            "comparison": {"type": "state_aware_greedy"},
            "solver": {"type": "torch_sbm", "constructorParameters": {"device": "cpu"},
                       "solverParameters": {"steps": 50, "runs": 8, "dtype": "float64"}},
            "executionPolicy": {"type": "batch", "batchSize": 2}}
        with tempfile.TemporaryDirectory() as directory:
            config = self.configuration(directory, calculator)
            path = Path(directory) / "prices.csv"
            lines = path.read_text().splitlines()
            second = [100, 103, 99, 100, 105, 101, 110, 105, 100, 109, 104]
            path.write_text(lines[0] + ",B\n" + "\n".join(f"{line},{price}" for line, price in zip(lines[1:], second)))
            config["portfolio"]["weights"] = {"A": 10, "B": -7}
            config["engine"]["riskStateGenerator"].update(
                type="correlated_returns_vola_grid", topKNeighbors=1, nZBins=3, nNearest=4)
            report = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
        self.assertAlmostEqual(report.margin, 0.42607412049256665, places=12)
        self.assertAlmostEqual(report.comparisonMargins["greedy"], 0.9927113937275229, places=12)

    def test_resident_execution_respects_a_one_problem_memory_budget(self):
        calculator = {"type": "bqm", "comparison": {"type": "state_aware_greedy"},
            "solver": {"type": "torch_sbm", "constructorParameters": {"device": "cpu"},
                       "solverParameters": {"steps": 8, "runs": 4, "dtype": "float64"}},
            "executionPolicy": {"type": "batch", "batchSize": 100, "maxBatchBytes": 1}}
        with tempfile.TemporaryDirectory() as directory:
            config = self.configuration(directory, calculator)
            report = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
        self.assertAlmostEqual(report.margin, 0.1798027685847499, places=12)
        self.assertEqual(report.numericalDiagnostics["scenarioCount"], 3)
        self.assertEqual(report.numericalDiagnostics["peakBatchProblems"], 1)
        self.assertEqual(report.numericalDiagnostics["fittedHostMaterializationBytes"], 8) # one float64 eigenvalue
        self.assertGreater(report.numericalDiagnostics["residentFitBytes"], 0)

    def test_unsupported_solver_is_rejected_while_loading_the_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.configuration(directory, {"type": "bqm", "solver": {"type": "random"}})
            with self.assertRaisesRegex(ValueError, "Torch solver"):
                MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory)

    def test_resident_greedy_rejects_nonfinite_scenario_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.configuration(directory)
            config["engine"]["riskStateGenerator"]["scenariosPerComponents"] = [101]
            path = Path(directory) / "prices.csv"
            path.write_text("date,A\n" + "\n".join(
                f"2024-01-{day:02d},{1e10 if day % 2 else 1.}" for day in range(1, 12)))
            with self.assertRaisesRegex(ValueError, "finite"):
                MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()

    def test_resident_solver_variants_and_batch_sizes_preserve_the_reference(self):
        for solver_type in ("torch_sbm", "adaptive_torch_sbm", "torch_svl"):
            for batch_size in (1, 4):
                with self.subTest(solver=solver_type, batch=batch_size), tempfile.TemporaryDirectory() as directory:
                    calculator = {"type": "bqm", "comparison": {"type": "state_aware_greedy"},
                        "solver": {"type": solver_type, "constructorParameters": {"device": "cpu"},
                                   "solverParameters": {"steps": 8, "runs": 4, "run_batch_size": 2, "dtype": "float64"}},
                        "executionPolicy": {"type": "batch", "batchSize": batch_size}}
                    config = self.configuration(directory, calculator)
                    report = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
                    # At eight steps the host SVL reference selects the less
                    # adverse feasible sample; a heuristic need not find the bound.
                    expected = 0. if solver_type == "torch_svl" else 0.1798027685847499
                    self.assertAlmostEqual(report.margin, expected, places=12)
                    self.assertAlmostEqual(report.comparisonMargins["greedy"], 0.1798027685847499, places=12)

    def test_resident_pca_matches_host_for_wide_inputs_and_canonical_asset_order(self):
        import numpy
        import pandas
        with tempfile.TemporaryDirectory() as directory:
            config = self.configuration(directory)
            rng = numpy.random.default_rng(512)
            names = [f"A{index:02d}" for index in range(16)]
            data = pandas.DataFrame(100 * numpy.exp(numpy.cumsum(rng.normal(0, .02, (11, 16)), axis=0)), columns=names)
            data.insert(0, "date", pandas.date_range("2024-01-01", periods=11))
            data.to_csv(Path(directory) / "prices.csv", index=False)
            config["portfolio"]["weights"] = {name: (-7 if index % 2 else 10) for index, name in reversed(list(enumerate(names)))}
            config["engine"]["riskStateGenerator"].update(components=2, scenariosPerComponents=[3, 3], nZBins=5)
            resident = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
            del config["engine"]["numericalExecution"]
            host = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
            self.assertAlmostEqual(resident.margin, host.margin, places=11)
            self.assertEqual(resident.numericalDiagnostics["scenarioCount"], 9)

    def test_gpu_resident_pipeline_matches_cpu_reference_when_available(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("requires a CUDA/ROCm GPU")
        calculator = {"type": "bqm", "comparison": {"type": "state_aware_greedy"},
            "solver": {"type": "torch_sbm", "constructorParameters": {"device": "cuda:0"},
                       "solverParameters": {"steps": 1, "runs": 4, "initial_scale": 0., "dtype": "float64"}},
            "executionPolicy": {"type": "batch", "batchSize": 2}}
        with tempfile.TemporaryDirectory() as directory:
            config = self.configuration(directory, calculator)
            config["engine"]["numericalExecution"]["device"] = "cuda:0"
            report = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
            self.assertAlmostEqual(report.margin, 0.1798027685847499, delta=2e-6)
            self.assertAlmostEqual(report.comparisonMargins["greedy"], 0.1798027685847499, delta=2e-6)
            self.assertEqual(report.numericalDiagnostics["dtype"], "float32")
            self.assertEqual(report.numericalDiagnostics["device"], "cuda:0")
            self.assertEqual(report.numericalDiagnostics["peakBatchProblems"], 2)

    def test_float32_risk_pipeline_preserves_margin_and_halves_fit_storage(self):
        import torch
        import numpy
        import pandas
        from unittest.mock import patch
        from margin_engine.torch_returns_execution import TorchReturnsExecution
        encode = TorchReturnsExecution._encode
        captured = []

        def capture(execution, weighted, grid, *args):
            for name in ("returns", "center", "nearest", "sigma", "means", "scales"):
                self.assertEqual(getattr(grid, name).dtype, torch.float32)
            result = encode(execution, weighted, grid, *args)
            self.assertEqual(result[0].linear.dtype, torch.float32)
            self.assertEqual(result[0].biases.dtype, torch.float32)
            self.assertEqual(result[0].source.linear.dtype, numpy.float64)
            captured.append(result[0])
            return result

        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            for correlated in (False, True):
                with self.subTest(device=device, correlated=correlated), tempfile.TemporaryDirectory() as directory:
                    config = self.configuration(directory)
                    rng = numpy.random.default_rng(512)
                    names = [f"A{index:02d}" for index in range(16)]
                    data = pandas.DataFrame(100 * numpy.exp(rng.normal(0, .02, (11, 16)).cumsum(axis=0)), columns=names)
                    data.insert(0, "date", pandas.date_range("2024-01-01", periods=11))
                    data.to_csv(Path(directory) / "prices.csv", index=False)
                    config["portfolio"]["weights"] = {name: (-7 if index % 2 else 10) for index, name in enumerate(names)}
                    config["engine"]["riskStateGenerator"].update(
                        type="correlated_returns_vola_grid" if correlated else "returns_vola_grid",
                        components=2, scenariosPerComponents=[3, 3], nZBins=5)
                    reference = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
                    config["engine"]["numericalExecution"].update(device=device, dtype="float32")
                    single = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
                    self.assertAlmostEqual(single.margin, reference.margin, delta=2e-5)
                    self.assertEqual(single.numericalDiagnostics["residentFitBytes"] * 2,
                                     reference.numericalDiagnostics["residentFitBytes"])
                    for solver in ("torch_sbm", "adaptive_torch_sbm", "torch_svl"):
                        config["engine"]["marginCalculator"] = {"type": "bqm", "comparison": {"type": "state_aware_greedy"},
                            "solver": {"type": solver, "constructorParameters": {"device": device},
                                       "solverParameters": {"steps": 8, "runs": 3, "run_batch_size": 2}},
                            "executionPolicy": {"type": "batch", "batchSize": 2}}
                        with patch.object(TorchReturnsExecution, "_encode", capture):
                            report = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
                        self.assertAlmostEqual(report.comparisonMargins["greedy"], reference.margin, delta=2e-5)
                        self.assertGreaterEqual(report.margin, 0.)
                        self.assertLessEqual(report.margin, report.comparisonMargins["greedy"] + 2e-5)
        self.assertTrue(captured)

    def test_yaml_precision_is_strict_and_reaches_the_typed_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.configuration(directory)
            for dtype in ("auto", "float32", "float64"):
                config["engine"]["numericalExecution"]["dtype"] = dtype
                config["engine"]["riskStateGenerator"]["pcaGridProvider"] = {
                    "backend": {"type": "torch", "device": "cpu", "dtype": dtype}}
                application = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory)
                self.assertEqual(application.engine.numericalExecution.dtype, dtype)
                self.assertEqual(application.engine.riskStateGenerator.pcaGridProvider.backend.dtype, dtype)
            for section in (config["engine"]["numericalExecution"],
                            config["engine"]["riskStateGenerator"]["pcaGridProvider"]["backend"]):
                section["dtype"] = "float16"
                with self.assertRaisesRegex(ValueError, "dtype"):
                    MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory)
                section["dtype"] = "auto"

    def test_fallback_bins_remain_auditable_in_resident_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.configuration(directory)
            config["engine"]["riskStateGenerator"].update(nZBins=6, residualSigmaRange=1e-12)
            report = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
            self.assertEqual(report.numericalDiagnostics["fallbackAssetCount"], 1)
            del config["engine"]["numericalExecution"]
            host = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
            self.assertAlmostEqual(report.margin, host.margin, places=12)

    def test_tied_correlation_nominations_use_canonical_asset_order(self):
        import numpy
        import pandas
        from risk_state_generator import RiskStateGenerationContext
        with tempfile.TemporaryDirectory() as directory:
            config = self.configuration(directory)
            rng = numpy.random.default_rng(3)
            names = [f"A{index}" for index in range(6)]
            data = pandas.DataFrame(100 * numpy.exp(numpy.cumsum(rng.normal(0, .02, (11, 6)), axis=0)), columns=names)
            data.insert(0, "date", pandas.date_range("2024-01-01", periods=11))
            data.to_csv(Path(directory) / "prices.csv", index=False)
            config["portfolio"]["weights"] = {name: 1 for name in names}
            config["engine"]["riskStateGenerator"].update(type="correlated_returns_vola_grid", nNearest=2, topKNeighbors=1)
            application = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory)
            generator = application.engine.riskStateGenerator.createRiskStateGenerator()
            request = generator.createDataRequest(application.portfolio, application.marginDate)
            state = next(generator.getRiskStates(RiskStateGenerationContext(data, request, application.marginDate)))
            pairs = set(zip(state.correlations.firstAssets, state.correlations.secondAssets))
            self.assertEqual(pairs, {(0, index) for index in range(1, 6)})

    def test_resident_mode_rejects_custom_encoding_instead_of_bypassing_it(self):
        from dataclasses import replace
        from margin_calculator.optimization.portfolio_risk_state_bqm_visitor import PortfolioRiskStateBQMVisitor
        class CustomVisitor(PortfolioRiskStateBQMVisitor):
            pass
        with tempfile.TemporaryDirectory() as directory:
            config = self.configuration(directory, {"type": "bqm", "solver": {
                "type": "torch_sbm", "constructorParameters": {"device": "cpu"}}})
            application = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory)
            calculator = replace(application.engine.marginCalculator, bqmVisitor=CustomVisitor())
            with self.assertRaisesRegex(ValueError, "custom"):
                replace(application.engine, marginCalculator=calculator)
