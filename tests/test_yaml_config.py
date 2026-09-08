"""Tests for complete YAML-driven margin application configuration."""

import tempfile
import unittest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import yaml

from data_manager import PartitionedPickleDataStore
from download_unit import LocalCSVDataProvider, ProductChunker
from margin_calculator import (
    BQMMarginCalculatorConfig,
    BatchBQMExecutionPolicy,
    GreedyMarginCalculatorConfig,
)
from margin_calculator.optimization.optimization_solver.bqm_solver import (
    TorchSBMBQMSolver,
    TorchSVLBQMSolver,
)
from margin_engine import MarginApplicationConfig, MarginReport
from risk_state_generator import (
    CorrelatedReturnsVolaGridRiskStateGeneratorConfig,
    ReturnsVolaGridRiskStateGeneratorConfig,
)


class YamlConfigurationTest(unittest.TestCase):
    def test_invalid_declarative_cache_limits_fail_during_parsing(self) -> None:
        for setting in (
            {"riskStateGenerator": {"pcaGridProvider": {"memorySize": 0}}},
            {"riskStateGenerator": {"pcaGridProvider": {"maxMemoryBytes": -1}}},
            {"marginCalculator": {"type": "bqm", "structuralCacheMemorySize": 0}},
        ):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                MarginApplicationConfig.fromYamlText(yaml.safe_dump({
                    "marginDate": "2024-01-11", "portfolio": {"weights": {"A": 1}},
                    "engine": setting,
                }), ".")


    def test_structural_cache_settings_are_declarative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump({
                "marginDate": "2024-01-11", "portfolio": {"weights": {"A": 1}},
                "engine": {"marginCalculator": {"type": "bqm", "structuralCacheMemorySize": 3}},
            }))
            config = MarginApplicationConfig.fromYaml(path).engine.marginCalculator
        self.assertIsNone(config.bqmVisitor)
        self.assertEqual(config.structuralCacheMemorySize, 3)

    def test_engines_from_one_config_do_not_reuse_fitted_market_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prices = root / "prices.csv"
            rows = [f"2024-01-{day:02d},{100 + day}" for day in range(1, 12)]
            prices.write_text("date,A\n" + "\n".join(rows))
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump({
                "marginDate": "2024-01-11", "portfolio": {"weights": {"A": 1}},
                "engine": {
                    "downloadManager": {"providers": {"local": "local_csv"},
                        "requestParameters": {"location": "prices.csv"}},
                    "riskStateGenerator": {"ew_window": 5, "nZBins": 1,
                        "scenariosPerComponents": [3], "allowEmptyBinFallback": True,
                        "pcaGridProvider": {"memorySize": 2}},
                    "marginCalculator": {"type": "greedy"},
                },
            }))
            application = MarginApplicationConfig.fromYaml(path)
            first = application.generateReport().margin
            rows[-2] = "2024-01-10,60"
            prices.write_text("date,A\n" + "\n".join(rows))
            second = application.generateReport().margin
            fresh = MarginApplicationConfig.fromYaml(path).generateReport().margin
        self.assertNotAlmostEqual(first, fresh)
        self.assertAlmostEqual(second, fresh)

    def test_rejects_unknown_download_retry_parameters_while_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump({
                "marginDate": "2024-01-11", "portfolio": {"weights": {"A": 1}},
                "engine": {"marginCalculator": {"type": "greedy"}, "downloadManager": {
                    "downloadAlgorithm": "exponential_backoff",
                    "downloadParameters": {"time": 0, "maxAttempt": 3,
                        "chunker": {"type": "date", "batchSize": 1}},
                }},
            }))
            with self.assertRaisesRegex(ValueError, "maxAttempt"):
                MarginApplicationConfig.fromYaml(path)

    def test_loads_a_wide_portfolio_and_capitalized_csv_dates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "portfolio.csv").write_text(
                "client_id,AAPL,MSFT\nclient,0.6,0.4\n",
                encoding="utf-8",
            )
            (root / "prices.csv").write_text(
                "Date,AAPL,MSFT\n2025-01-02,10,20\n2025-01-03,11,21\n",
                encoding="utf-8",
            )
            config_path = root / "margin.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2025-01-02",
                        "portfolio": {
                            "csv": "portfolio.csv",
                            "clientId": "client",
                        },
                        "backtest": {
                            "datesFromCsv": {"path": "prices.csv"},
                        },
                        "engine": {"marginCalculator": {"type": "greedy"}},
                    }
                ),
                encoding="utf-8",
            )

            application = MarginApplicationConfig.fromYaml(config_path)

        self.assertEqual(application.portfolio.weights["AAPL"], Decimal("0.6"))
        self.assertEqual(
            application.backtestRequests["default"].dates,
            (date(2025, 1, 2), date(2025, 1, 3)),
        )

    def test_long_portfolio_csv_sums_duplicate_instrument_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "portfolio.csv").write_text(
                "client_id,ticker,weight\nclient,AAPL,1.25\nclient,AAPL,-0.25\n",
                encoding="utf-8",
            )
            config_path = root / "margin.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2025-01-02",
                        "portfolio": {
                            "csv": "portfolio.csv",
                            "clientId": "client",
                        },
                        "engine": {"marginCalculator": {"type": "greedy"}},
                    }
                ),
                encoding="utf-8",
            )

            application = MarginApplicationConfig.fromYaml(config_path)

        self.assertEqual(application.portfolio.weights["AAPL"], Decimal("1.00"))

    def test_loads_and_runs_a_complete_local_greedy_application(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prices = root / "prices.csv"
            rows = ["date,AAPL"]
            for day in range(11):
                current = date(2024, 1, 1) + timedelta(days=day)
                rows.append(f"{current.isoformat()},{100 + day}")
            prices.write_text("\n".join(rows), encoding="utf-8")
            config_path = root / "margin.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2024-01-11",
                        "portfolio": {
                            "weights": {"AAPL": "10"},
                            "cash": "5",
                        },
                        "engine": {
                            "downloadManager": {
                                "providers": {"local": "local_csv"},
                                "requestParameters": {
                                    "location": "prices.csv"
                                },
                            },
                            "dataManager": {
                                "memorySize": 2,
                                "backingStore": {
                                    "type": "partitioned_pickle",
                                    "directory": "cache",
                                },
                            },
                            "riskStateGenerator": {
                                "type": "returns_vola_grid",
                                "ew_window": 5,
                                "components": 1,
                                "scenariosPerComponents": [1],
                                "nZBins": 3,
                            },
                            "marginCalculator": {"type": "greedy"},
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            application = MarginApplicationConfig.fromYaml(config_path)
            report = application.generateReport()

            self.assertEqual(application.marginDate, date(2024, 1, 11))
            self.assertEqual(application.portfolio.cash, 5)
            self.assertIsInstance(
                application.engine.marginCalculator,
                GreedyMarginCalculatorConfig,
            )
            self.assertIsInstance(
                application.engine.riskStateGenerator,
                ReturnsVolaGridRiskStateGeneratorConfig,
            )
            self.assertIsInstance(
                next(iter(application.engine.downloadManager.providers.values())),
                LocalCSVDataProvider,
            )
            self.assertIsInstance(
                application.engine.dataManager.backingStore,
                PartitionedPickleDataStore,
            )
            self.assertEqual(
                application.engine.downloadManager.requestParameters["location"],
                str(prices),
            )
            self.assertIsInstance(report, MarginReport)

    def test_loads_the_complete_example_bqm_configuration(self) -> None:
        application = MarginApplicationConfig.fromYaml(
            "config/margin.example.yaml"
        )

        calculator = application.engine.marginCalculator
        generator = application.engine.riskStateGenerator
        self.assertIsInstance(calculator, BQMMarginCalculatorConfig)
        self.assertEqual(calculator.solver.solverType, "torch_svl")
        self.assertEqual(
            calculator.solver.constructorParameters,
            {"device": "auto"},
        )
        self.assertEqual(calculator.solver.solverParameters["runs"], 64)
        self.assertEqual(
            calculator.solver.solverParameters["integrator"],
            "weak_order_2",
        )
        self.assertEqual(calculator.comparisonPnlAnchor, "market")
        self.assertEqual(
            calculator.solver.solverParameters["dtype"],
            "float32",
        )
        self.assertIsInstance(calculator.executionPolicy, BatchBQMExecutionPolicy)
        self.assertEqual(calculator.executionPolicy.batchSize, 105)
        self.assertEqual(calculator.executionPolicy.maxBatchBytes, 536870912)
        self.assertIsInstance(
            generator,
            CorrelatedReturnsVolaGridRiskStateGeneratorConfig,
        )
        self.assertEqual(generator.components, 2)
        self.assertEqual(generator.scenariosPerComponents, (21, 5))
        self.assertEqual(generator.nZBins, 21)
        self.assertEqual(
            application.backtestOutputDirectory,
            Path("backtest_results/example").resolve(),
        )
        self.assertEqual(
            application.backtestRequests["default"].dates,
            (
                date(2024, 1, 29),
                date(2024, 1, 30),
                date(2024, 1, 31),
            ),
        )

    def test_builds_nested_download_chunkers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2024-01-11",
                        "portfolio": {"weights": {"AAPL": 1}},
                        "engine": {
                            "downloadManager": {
                                "providers": {"yahoo": "yfinance"},
                                "downloadAlgorithm": "exponential_backoff",
                                "downloadParameters": {
                                    "time": 0,
                                    "chunker": {
                                        "type": "product",
                                        "first": {
                                            "type": "instrument",
                                            "batchSize": 5,
                                        },
                                        "second": {
                                            "type": "date",
                                            "batchSize": 30,
                                        },
                                    },
                                },
                            },
                            "marginCalculator": {"type": "greedy"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            application = MarginApplicationConfig.fromYaml(path)

        self.assertIsInstance(
            application.engine.downloadManager.downloadParameters["chunker"],
            ProductChunker,
        )

    def test_constructs_torch_sbm_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torch.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2024-01-11",
                        "portfolio": {"weights": {"AAPL": 1}},
                        "engine": {
                            "marginCalculator": {
                                "type": "bqm",
                                "solver": {
                                    "type": "torch_sbm",
                                    "constructorParameters": {"device": "cpu"},
                                    "solverParameters": {
                                        "steps": 25,
                                        "runs": 4,
                                        "dtype": "float64",
                                    },
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            application = MarginApplicationConfig.fromYaml(path)

        calculator = application.engine.marginCalculator
        self.assertIsInstance(calculator, BQMMarginCalculatorConfig)
        solver = calculator.solver.createBQMSolver()
        self.assertIsInstance(solver, TorchSBMBQMSolver)
        self.assertEqual(solver.device, "cpu")
        self.assertEqual(calculator.solver.solverParameters["runs"], 4)

    def test_constructs_multi_device_torch_sbm_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torch_multi_gpu.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2024-01-11",
                        "portfolio": {"weights": {"AAPL": 1}},
                        "engine": {
                            "marginCalculator": {
                                "type": "bqm",
                                "solver": {
                                    "type": "torch_sbm",
                                    "constructorParameters": {
                                        "devices": ["cuda:0", "cuda:1"]
                                    },
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            application = MarginApplicationConfig.fromYaml(path)

        calculator = application.engine.marginCalculator
        self.assertIsInstance(calculator, BQMMarginCalculatorConfig)
        solver = calculator.solver.createBQMSolver()
        self.assertIsInstance(solver, TorchSBMBQMSolver)
        self.assertEqual(solver.requestedDevices, ("cuda:0", "cuda:1"))

    def test_constructs_multi_device_torch_svl_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torch_svl.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2024-01-11",
                        "portfolio": {"weights": {"AAPL": 1}},
                        "engine": {
                            "marginCalculator": {
                                "type": "bqm",
                                "solver": {
                                    "type": "torch_svl",
                                    "constructorParameters": {
                                        "devices": ["cuda:0", "cuda:1"]
                                    },
                                    "solverParameters": {
                                        "steps": 25,
                                        "runs": 4,
                                        "integrator": "weak_order_2",
                                    },
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            application = MarginApplicationConfig.fromYaml(path)

        calculator = application.engine.marginCalculator
        self.assertIsInstance(calculator, BQMMarginCalculatorConfig)
        solver = calculator.solver.createBQMSolver()
        self.assertIsInstance(solver, TorchSVLBQMSolver)
        self.assertEqual(solver.requestedDevices, ("cuda:0", "cuda:1"))
        self.assertEqual(
            calculator.solver.solverParameters["integrator"], "weak_order_2"
        )

    def test_rejects_unknown_yaml_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2024-01-11",
                        "portfolio": {"weights": {"AAPL": 1}},
                        "engine": {"unknown": True},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unknown YAML keys"):
                MarginApplicationConfig.fromYaml(path)


if __name__ == "__main__":
    unittest.main()
